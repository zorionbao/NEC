from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA

from revision_stats import (
    RANDOM_SEED,
    bh_fdr,
    bootstrap_hedges_g,
    exact_matrix_permutation,
    hedges_g,
    paired_sign_flip,
    safe_zscore,
)


TARGET_METABOLITES = {
    "Succinate": "succinic acid",
    "Lactate": "lactic acid",
    "Inosine": "inosine",
    "Nicotinic acid": "nicotinic acid",
    "Indole-3-lactate": "indole-3-lactate",
    "Indole-3-acetate": "indole-3-acetate",
    "3-Hydroxybutyrate": "3-hydroxybutyric acid",
    "Chenodeoxycholate": "chenodeoxycholic acid",
    "3-Hydroxypropionate": "3-hydroxypropionic acid",
}

PATHWAY_PATTERNS = {
    "TCA cycle": r"\bTCA\b|tricarboxylic|citrate cycle",
    "Glyoxylate metabolism": r"glyoxylate",
    "Pyruvate metabolism": r"pyruvate",
    "Propanoate metabolism": r"propanoate|propionate",
    "Succinate metabolism": r"succinate|succinyl",
    "Tryptophan metabolism": r"tryptophan|indole",
    "Purine metabolism": r"purine|inosine|adenosine|guanosine",
    "Nicotinate metabolism": (
        r"nicotinate|nicotinamide|NAD[- ]BIOSYNTHESIS|"
        r"NAD biosynthesis|NAD salvage"
    ),
}

IMPLAUSIBLE_PATHWAY_PATTERN = re.compile(
    r"engineered|saccharomyces|yeast|plants?|photosynthetic|"
    r"calvin|eukaryot|mammals?|chloroplast",
    flags=re.IGNORECASE,
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


def clr_transform(matrix: np.ndarray, pseudocount: float | None = None) -> tuple[np.ndarray, float]:
    values = np.asarray(matrix, dtype=float)
    if (values < 0).any():
        raise ValueError("CLR input contains negative values")
    positive = values[values > 0]
    if pseudocount is None:
        pseudocount = float(positive.min() / 2.0) if positive.size else 1e-8
    logged = np.log(values + pseudocount)
    return logged - logged.mean(axis=1, keepdims=True), pseudocount


def vector_hedges_g(matrix: np.ndarray, labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels, dtype=object)
    case = matrix[labels == "nec"]
    control = matrix[labels == "control"]
    output = np.empty(matrix.shape[1], dtype=float)
    for index in range(matrix.shape[1]):
        output[index] = hedges_g(case[:, index], control[:, index])
    return output


def add_selected_confidence_intervals(
    effects: pd.DataFrame,
    matrix: pd.DataFrame,
    labels: pd.Series,
    selected_features: list[str],
    feature_column: str,
    random_seed_offset: int,
) -> pd.DataFrame:
    output = effects.copy()
    output["hedges_g_ci_low"] = np.nan
    output["hedges_g_ci_high"] = np.nan
    for index, feature in enumerate(selected_features):
        if feature not in matrix.columns:
            continue
        case = matrix.loc[labels.eq("nec"), feature]
        control = matrix.loc[labels.eq("control"), feature]
        low, high = bootstrap_hedges_g(
            case,
            control,
            n_bootstrap=5000,
            random_seed=RANDOM_SEED + random_seed_offset + index,
        )
        row_mask = output[feature_column].eq(feature)
        output.loc[row_mask, "hedges_g_ci_low"] = low
        output.loc[row_mask, "hedges_g_ci_high"] = high
    return output


def exact_prevalence_test(
    presence: pd.DataFrame,
    labels: pd.Series,
) -> tuple[np.ndarray, np.ndarray]:
    case = presence.loc[labels.eq("nec")]
    control = presence.loc[labels.eq("control")]
    delta = case.mean(axis=0).to_numpy() - control.mean(axis=0).to_numpy()
    p_values = np.empty(presence.shape[1], dtype=float)
    for index, feature in enumerate(presence.columns):
        table = np.array(
            [
                [
                    int(case[feature].sum()),
                    int(len(case) - case[feature].sum()),
                ],
                [
                    int(control[feature].sum()),
                    int(len(control) - control[feature].sum()),
                ],
            ]
        )
        p_values[index] = stats.fisher_exact(table, alternative="two-sided").pvalue
    return delta, p_values


def derive_infant_id(isolate: Any) -> str:
    value = str(isolate).strip()
    match = re.search(
        r"(?:(NEC|Control)-[DR]?|NOR)(\d+)$",
        value,
        flags=re.IGNORECASE,
    )
    if match is None:
        raise ValueError(f"Could not derive infant identifier from isolate {value!r}")
    prefix = (
        "NEC"
        if match.group(1) is not None and match.group(1).lower() == "nec"
        else "Control"
    )
    return f"{prefix}-{int(match.group(2))}"


def add_run_metadata(
    matrix: pd.DataFrame,
    selected_runs: pd.DataFrame,
) -> pd.DataFrame:
    metadata = selected_runs[["run", "isolate", "condition"]].copy()
    metadata["infant_id"] = metadata["isolate"].map(derive_infant_id)
    metadata["clinical_group"] = np.select(
        [
            metadata["isolate"].astype(str).str.startswith("NEC-D"),
            metadata["isolate"].astype(str).str.startswith("NEC-R"),
        ],
        ["NEC-D", "NEC-R"],
        default="Control",
    )
    metadata = metadata.set_index("run").loc[matrix.index]
    if metadata.groupby("infant_id")["condition"].nunique().max() != 1:
        raise ValueError("An infant maps to more than one NEC/control condition")
    return metadata


def aggregate_runs_to_infants(
    matrix: pd.DataFrame,
    run_metadata: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    work = matrix.copy()
    work.insert(0, "infant_id", run_metadata.loc[work.index, "infant_id"])
    infant_matrix = work.groupby("infant_id", sort=True).mean(numeric_only=True)
    infant_labels = (
        run_metadata.reset_index()
        .drop_duplicates(["infant_id", "condition"])
        .set_index("infant_id")["condition"]
        .reindex(infant_matrix.index)
        .astype(str)
    )
    return infant_matrix, infant_labels


def paired_phase_effects(
    matrix: pd.DataFrame,
    run_metadata: pd.DataFrame,
    feature_column: str,
) -> pd.DataFrame:
    metadata = run_metadata.reset_index()
    paired_ids = (
        metadata.loc[metadata["clinical_group"].isin(["NEC-D", "NEC-R"])]
        .groupby("infant_id")["clinical_group"]
        .nunique()
        .loc[lambda values: values.eq(2)]
        .index
    )
    rows: list[dict[str, Any]] = []
    for feature in matrix.columns:
        deltas = []
        for infant_id in paired_ids:
            infant = metadata.loc[metadata["infant_id"].eq(infant_id)]
            acute_run = infant.loc[infant["clinical_group"].eq("NEC-D"), "run"].iloc[0]
            recovery_run = infant.loc[infant["clinical_group"].eq("NEC-R"), "run"].iloc[0]
            deltas.append(float(matrix.loc[acute_run, feature] - matrix.loc[recovery_run, feature]))
        test = paired_sign_flip(deltas)
        standard_deviation = float(np.std(deltas, ddof=1)) if len(deltas) > 1 else float("nan")
        rows.append(
            {
                feature_column: feature,
                "mean_delta_acute_minus_recovery": test["mean_delta"],
                "paired_cohen_dz": (
                    float(np.mean(deltas) / standard_deviation)
                    if np.isfinite(standard_deviation) and standard_deviation > 0
                    else float("nan")
                ),
                "p_value": test["p_value"],
                "n_pairs": test["n_pairs"],
                "permutation_method": test["method"],
            }
        )
    output = pd.DataFrame(rows)
    output["fdr_bh"] = bh_fdr(output["p_value"])
    return output.sort_values(["fdr_bh", "p_value"])


def profile_rarefaction_sensitivity(
    abundance_percent: pd.DataFrame,
    run_metadata: pd.DataFrame,
    output_dir: Path,
) -> pd.DataFrame:
    observed_fraction = abundance_percent.div(100.0)
    observed_clr_values, _ = clr_transform(observed_fraction.to_numpy())
    observed_clr = pd.DataFrame(
        observed_clr_values,
        index=observed_fraction.index,
        columns=observed_fraction.columns,
    )
    observed_infant_clr, observed_labels = aggregate_runs_to_infants(
        observed_clr,
        run_metadata,
    )
    observed_delta = (
        observed_infant_clr.loc[observed_labels.eq("nec")].mean(axis=0)
        - observed_infant_clr.loc[observed_labels.eq("control")].mean(axis=0)
    )
    infant_ids = observed_infant_clr.index.tolist()
    infant_membership = np.zeros(
        (len(infant_ids), len(observed_fraction)),
        dtype=float,
    )
    run_infants = run_metadata.loc[observed_fraction.index, "infant_id"]
    for infant_index, infant_id in enumerate(infant_ids):
        run_indices = np.flatnonzero(run_infants.to_numpy() == infant_id)
        infant_membership[infant_index, run_indices] = 1.0 / len(run_indices)
    infant_conditions = observed_labels.reindex(infant_ids).to_numpy()
    nec_mask = infant_conditions == "nec"
    control_mask = infant_conditions == "control"
    normalized_probabilities = observed_fraction.to_numpy(float)
    row_sums = normalized_probabilities.sum(axis=1, keepdims=True)
    normalized_probabilities = np.divide(
        normalized_probabilities,
        row_sums,
        out=np.full_like(
            normalized_probabilities,
            1.0 / normalized_probabilities.shape[1],
        ),
        where=row_sums > 0,
    )

    rng = np.random.default_rng(RANDOM_SEED + 4500)
    rows = []
    for depth in (500, 1_000, 5_000, 10_000, 50_000, 100_000, 200_000):
        correlations = []
        sign_agreements = []
        for _ in range(250):
            simulated = np.vstack(
                [
                    rng.multinomial(depth, probabilities) / depth
                    for probabilities in normalized_probabilities
                ]
            )
            simulated_clr_values, _ = clr_transform(
                simulated,
                pseudocount=0.5 / depth,
            )
            simulated_infant_clr = infant_membership @ simulated_clr_values
            simulated_delta = pd.Series(
                simulated_infant_clr[nec_mask].mean(axis=0)
                - simulated_infant_clr[control_mask].mean(axis=0),
                index=observed_fraction.columns,
            )
            correlation = stats.spearmanr(
                observed_delta,
                simulated_delta,
            ).correlation
            correlations.append(float(correlation))
            sign_agreements.append(
                float(
                    (
                        np.sign(observed_delta.to_numpy())
                        == np.sign(simulated_delta.to_numpy())
                    ).mean()
                )
            )
        correlation_interval = np.quantile(correlations, [0.025, 0.5, 0.975])
        sign_interval = np.quantile(sign_agreements, [0.025, 0.5, 0.975])
        rows.append(
            {
                "depth": depth,
                "spearman_to_200k_profile_delta": correlation_interval[1],
                "spearman_ci_low": correlation_interval[0],
                "spearman_ci_high": correlation_interval[2],
                "delta_sign_stability": sign_interval[1],
                "sign_stability_ci_low": sign_interval[0],
                "sign_stability_ci_high": sign_interval[2],
                "n_species": observed_fraction.shape[1],
                "n_simulations": 250,
                "analysis_unit": "infant mean across available runs",
                "reference_profile": "observed 200,000-spot MetaPhlAn profile",
            }
        )
    output = pd.DataFrame(rows)
    output.to_csv(
        output_dir / "metaphlan_profile_rarefaction_sensitivity_corrected.csv",
        index=False,
    )
    return output


def microbiome_species_analysis(
    snapshot_results: Path,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    directory = snapshot_results / "microbiome_strict" / "metaphlan_native_subset"
    species = pd.read_csv(directory / "metaphlan_species_abundance_matrix.csv")
    selected_runs = pd.read_csv(directory / "selected_runs.csv")
    feature_columns = [
        column for column in species.columns if column not in {"run", "condition"}
    ]
    abundance_percent = species.set_index("run")[feature_columns].astype(float)
    labels = species.set_index("run")["condition"].astype(str)
    abundance_fraction = abundance_percent / 100.0
    clr_values, pseudocount = clr_transform(abundance_fraction.to_numpy())
    clr = pd.DataFrame(
        clr_values,
        index=abundance_fraction.index,
        columns=feature_columns,
    )
    run_metadata = add_run_metadata(clr, selected_runs)
    infant_clr, infant_labels = aggregate_runs_to_infants(clr, run_metadata)
    infant_abundance, _ = aggregate_runs_to_infants(
        abundance_percent,
        run_metadata,
    )
    observed, p_values, n_permutations, method = exact_matrix_permutation(
        infant_clr.to_numpy(),
        infant_labels,
    )
    g_values = vector_hedges_g(infant_clr.to_numpy(), infant_labels.to_numpy())
    prevalence = infant_abundance.gt(0)
    prevalence_delta, prevalence_p = exact_prevalence_test(
        prevalence,
        infant_labels.reindex(prevalence.index),
    )
    effects = pd.DataFrame(
        {
            "species": feature_columns,
            "mean_clr_delta_nec_vs_control": observed,
            "hedges_g": g_values,
            "p_value": p_values,
            "fdr_bh": bh_fdr(p_values),
            "prevalence_delta_nec_vs_control": prevalence_delta,
            "prevalence_p_value": prevalence_p,
            "prevalence_fdr_bh": bh_fdr(prevalence_p),
            "mean_abundance_nec_percent": infant_abundance.loc[
                infant_labels.eq("nec")
            ].mean(axis=0).to_numpy(),
            "mean_abundance_control_percent": infant_abundance.loc[
                infant_labels.eq("control")
            ].mean(axis=0).to_numpy(),
            "prevalence_nec": prevalence.loc[
                infant_labels.eq("nec")
            ].mean(axis=0).to_numpy(),
            "prevalence_control": prevalence.loc[
                infant_labels.eq("control")
            ].mean(axis=0).to_numpy(),
            "n_nec": int(infant_labels.eq("nec").sum()),
            "n_control": int(infant_labels.eq("control").sum()),
            "analysis_unit": "infant mean across available runs",
            "n_permutations": n_permutations,
            "permutation_method": method,
        }
    )
    selected_for_ci = effects["species"].tolist()
    effects = add_selected_confidence_intervals(
        effects,
        infant_clr,
        infant_labels,
        selected_for_ci,
        feature_column="species",
        random_seed_offset=500,
    )
    effects = effects.sort_values(
        ["fdr_bh", "p_value", "hedges_g"],
        ascending=[True, True, False],
    )
    effects.to_csv(output_dir / "metaphlan_species_effects_corrected.csv", index=False)

    sample_long = (
        abundance_percent.reset_index()
        .melt(id_vars="run", var_name="species", value_name="abundance_percent")
        .merge(species[["run", "condition"]], on="run", how="left")
        .merge(
            run_metadata.reset_index()[
                ["run", "isolate", "infant_id", "clinical_group"]
            ],
            on="run",
            how="left",
        )
    )
    sample_long = sample_long.merge(
        clr.reset_index().melt(
            id_vars="run",
            var_name="species",
            value_name="clr_abundance",
        ),
        on=["run", "species"],
        how="left",
    )
    sample_long.to_csv(output_dir / "metaphlan_species_sample_long_corrected.csv", index=False)
    infant_clr.reset_index().merge(
        infant_labels.rename("condition"),
        left_on="infant_id",
        right_index=True,
    ).to_csv(
        output_dir / "metaphlan_species_infant_clr_matrix.csv",
        index=False,
    )
    phase_effects = paired_phase_effects(clr, run_metadata, "species")
    phase_effects.to_csv(
        output_dir / "metaphlan_species_nec_phase_paired_effects.csv",
        index=False,
    )
    profile_rarefaction_sensitivity(
        abundance_percent,
        run_metadata,
        output_dir,
    )

    pca = PCA(n_components=2, random_state=RANDOM_SEED)
    coordinates = pca.fit_transform(clr)
    pca_frame = species[["run", "condition"]].copy()
    pca_frame["aitchison_pc1"] = coordinates[:, 0]
    pca_frame["aitchison_pc2"] = coordinates[:, 1]
    pca_frame = pca_frame.merge(
        selected_runs[["run", "isolate"]],
        on="run",
        how="left",
    )
    pca_frame["clinical_group"] = np.select(
        [
            pca_frame["isolate"].astype(str).str.startswith("NEC-D"),
            pca_frame["isolate"].astype(str).str.startswith("NEC-R"),
        ],
        ["NEC-D", "NEC-R"],
        default="Control",
    )
    pca_frame.to_csv(output_dir / "metaphlan_species_aitchison_pca.csv", index=False)
    loadings = pd.DataFrame(
        {
            "species": feature_columns,
            "pc1_loading": pca.components_[0],
            "pc2_loading": pca.components_[1],
        }
    )
    loadings.to_csv(output_dir / "metaphlan_species_aitchison_loadings.csv", index=False)
    write_json(
        output_dir / "metaphlan_species_transform.json",
        {
            "pseudocount_fraction": pseudocount,
            "input_depth_label": "MetaPhlAn profile from 200,000-spot bounded subset per run",
            "n_runs": len(species),
            "n_infants": int(infant_clr.shape[0]),
            "inferential_unit": "infant mean across available runs",
            "n_species": len(feature_columns),
            "pc1_variance_fraction": pca.explained_variance_ratio_[0],
            "pc2_variance_fraction": pca.explained_variance_ratio_[1],
        },
    )
    return effects, sample_long, clr, pca_frame


def pathway_category(pathway: str) -> str | None:
    for category, pattern in PATHWAY_PATTERNS.items():
        if re.search(pattern, pathway, flags=re.IGNORECASE):
            return category
    return None


def plausible_microbial_pathway(pathway: str) -> bool:
    return IMPLAUSIBLE_PATHWAY_PATTERN.search(str(pathway)) is None


def humann_pathway_analysis(
    snapshot_results: Path,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    directory = snapshot_results / "microbiome_strict" / "humann_native_subset"
    pathways = pd.read_csv(directory / "humann_pathabundance_matrix.csv")
    feature_columns = [
        column
        for column in pathways.columns
        if column not in {"run", "condition"}
        and not column.upper().startswith("UNINTEGRATED")
        and not column.upper().startswith("UNMAPPED")
        and "|" not in column
    ]
    raw = pathways.set_index("run")[feature_columns].astype(float)
    mapped_total = raw.sum(axis=1).replace(0, np.nan)
    cpm = raw.div(mapped_total, axis=0) * 1_000_000.0
    cpm = cpm.fillna(0.0)
    log_cpm = np.log1p(cpm)
    labels = pathways.set_index("run")["condition"].astype(str)
    selected_runs = pd.read_csv(
        snapshot_results
        / "microbiome_strict"
        / "metaphlan_native_subset"
        / "selected_runs.csv"
    )
    run_metadata = add_run_metadata(log_cpm, selected_runs)
    infant_log_cpm, infant_labels = aggregate_runs_to_infants(
        log_cpm,
        run_metadata,
    )
    infant_cpm, _ = aggregate_runs_to_infants(cpm, run_metadata)
    observed, p_values, n_permutations, method = exact_matrix_permutation(
        infant_log_cpm.to_numpy(),
        infant_labels,
    )
    g_values = vector_hedges_g(
        infant_log_cpm.to_numpy(),
        infant_labels.to_numpy(),
    )
    effects = pd.DataFrame(
        {
            "pathway": feature_columns,
            "log1p_cpm_delta_nec_vs_control": observed,
            "hedges_g": g_values,
            "p_value": p_values,
            "fdr_bh": bh_fdr(p_values),
            "mean_cpm_nec": infant_cpm.loc[
                infant_labels.eq("nec")
            ].mean(axis=0).to_numpy(),
            "mean_cpm_control": infant_cpm.loc[
                infant_labels.eq("control")
            ].mean(axis=0).to_numpy(),
            "n_nec": int(infant_labels.eq("nec").sum()),
            "n_control": int(infant_labels.eq("control").sum()),
            "analysis_unit": "infant mean across available runs",
            "n_permutations": n_permutations,
            "permutation_method": method,
        }
    )
    effects["pathway_category"] = effects["pathway"].map(pathway_category)
    effects["plausible_microbial_pathway"] = effects["pathway"].map(
        plausible_microbial_pathway
    )
    effects[["pathway", "plausible_microbial_pathway"]].assign(
        exclusion_reason=lambda frame: np.where(
            frame["plausible_microbial_pathway"],
            "",
            "engineered or non-microbial host-context pathway label",
        )
    ).to_csv(
        output_dir / "humann_pathway_biological_filter_inventory.csv",
        index=False,
    )
    plausible_effects = effects.loc[effects["plausible_microbial_pathway"]].copy()
    targeted = (
        plausible_effects.dropna(subset=["pathway_category"])
        .sort_values(["pathway_category", "p_value", "hedges_g"], ascending=[True, True, False])
        .groupby("pathway_category", as_index=False)
        .head(2)
    )
    top = plausible_effects.reindex(
        plausible_effects["hedges_g"].abs().sort_values(ascending=False).index
    ).head(12)
    selected_features = list(dict.fromkeys(targeted["pathway"].tolist() + top["pathway"].tolist()))
    effects = add_selected_confidence_intervals(
        effects,
        infant_log_cpm,
        infant_labels,
        selected_features,
        feature_column="pathway",
        random_seed_offset=600,
    )
    effects = effects.sort_values(["fdr_bh", "p_value", "hedges_g"], ascending=[True, True, False])
    effects.to_csv(output_dir / "humann_pathway_effects_corrected.csv", index=False)
    selected = effects.loc[effects["pathway"].isin(selected_features)].copy()
    selected["selection_reason"] = np.where(
        selected["pathway"].isin(targeted["pathway"]),
        "targeted_metabolic_axis",
        "largest_absolute_effect",
    )
    selected.to_csv(output_dir / "humann_pathway_selected_panel.csv", index=False)

    sample_long = (
        log_cpm[selected_features]
        .reset_index()
        .melt(id_vars="run", var_name="pathway", value_name="log1p_cpm")
        .merge(pathways[["run", "condition"]], on="run", how="left")
        .merge(
            run_metadata.reset_index()[
                ["run", "isolate", "infant_id", "clinical_group"]
            ],
            on="run",
            how="left",
        )
    )
    sample_long.to_csv(output_dir / "humann_pathway_selected_sample_long.csv", index=False)
    cpm.reset_index().to_csv(output_dir / "humann_pathway_cpm_matrix.csv.gz", index=False, compression="gzip")
    log_cpm.reset_index().to_csv(
        output_dir / "humann_pathway_log1p_cpm_matrix.csv.gz",
        index=False,
        compression="gzip",
    )
    infant_log_cpm.reset_index().merge(
        infant_labels.rename("condition"),
        left_on="infant_id",
        right_index=True,
    ).to_csv(
        output_dir / "humann_pathway_infant_log1p_cpm_matrix.csv.gz",
        index=False,
        compression="gzip",
    )
    phase_effects = paired_phase_effects(log_cpm, run_metadata, "pathway")
    phase_effects.to_csv(
        output_dir / "humann_pathway_nec_phase_paired_effects.csv",
        index=False,
    )
    return effects, selected, log_cpm, sample_long


def normalize_metabolite_name(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value).strip().lower())


def metabolome_patient_analysis(
    snapshot_results: Path,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    directory = snapshot_results / "microbiome_metabolome"
    metadata = pd.read_csv(directory / "infant_metadata_harmonized.csv")
    matrix = pd.read_csv(directory / "metabolite_matrix.csv")
    metadata["sample_key"] = metadata["sample"].astype(str)
    matrix = matrix.rename(columns={matrix.columns[0]: "feature"})
    matrix["feature"] = matrix["feature"].astype(str)
    sample_columns = [
        column for column in matrix.columns[1:] if column in set(metadata["sample_key"])
    ]
    sample_metadata = metadata.set_index("sample_key").loc[sample_columns].copy()
    eligible = (
        sample_metadata["Group"].astype(str).isin(["control", "nec"])
        & ~sample_metadata["Group"].astype(str).eq("GI")
    )
    sample_columns = sample_metadata.index[eligible].tolist()
    sample_metadata = sample_metadata.loc[sample_columns].copy()
    values = matrix.set_index("feature")[sample_columns].T.astype(float)
    values = values.loc[:, ~values.columns.astype(str).str.fullmatch("mTIC", case=False)]
    minimum = float(np.nanmin(values.to_numpy()))
    if minimum < 0:
        raise ValueError("Metabolite intensity matrix contains negative values")
    log_values = np.log1p(values)
    log_values["Patient"] = sample_metadata["Patient"].to_numpy()
    log_values["condition"] = sample_metadata["Group"].astype(str).to_numpy()
    patient = (
        log_values.groupby(["Patient", "condition"], as_index=False)
        .mean(numeric_only=True)
        .set_index(["Patient", "condition"])
    )
    labels = pd.Series(
        patient.index.get_level_values("condition"),
        index=patient.index,
        name="condition",
    )
    observed, p_values, n_permutations, method = exact_matrix_permutation(
        patient.to_numpy(),
        labels,
    )
    g_values = vector_hedges_g(patient.to_numpy(), labels.to_numpy())
    feature_names = patient.columns.astype(str).tolist()
    effects = pd.DataFrame(
        {
            "feature": feature_names,
            "named_feature": [
                not bool(re.fullmatch(r"\d+(?:\.\d+)?", feature.strip()))
                for feature in feature_names
            ],
            "patient_level_log_delta_nec_vs_control": observed,
            "hedges_g": g_values,
            "p_value": p_values,
            "fdr_bh": bh_fdr(p_values),
            "n_nec_patients": int(labels.eq("nec").sum()),
            "n_control_patients": int(labels.eq("control").sum()),
            "n_permutations": n_permutations,
            "permutation_method": method,
        }
    )
    normalized_to_feature = {
        normalize_metabolite_name(feature): feature for feature in feature_names
    }
    target_rows = []
    selected_features = []
    for canonical, exact_name in TARGET_METABOLITES.items():
        observed_name = normalized_to_feature.get(normalize_metabolite_name(exact_name))
        target_rows.append(
            {
                "canonical_metabolite": canonical,
                "exact_required_name": exact_name,
                "observed_feature": observed_name,
                "present": observed_name is not None,
                "annotation_rule": "case-insensitive exact name match",
            }
        )
        if observed_name is not None:
            selected_features.append(observed_name)
    target_inventory = pd.DataFrame(target_rows)
    target_inventory.to_csv(
        output_dir / "target_metabolite_exact_match_inventory.csv",
        index=False,
    )
    effects = add_selected_confidence_intervals(
        effects,
        patient.reset_index(drop=True),
        labels.reset_index(drop=True),
        selected_features,
        feature_column="feature",
        random_seed_offset=700,
    )
    effects = effects.sort_values(["fdr_bh", "p_value", "hedges_g"], ascending=[True, True, False])
    effects.to_csv(output_dir / "metabolite_patient_level_effects.csv", index=False)

    targeted_effects = target_inventory.merge(
        effects,
        left_on="observed_feature",
        right_on="feature",
        how="left",
    )
    targeted_effects.to_csv(output_dir / "target_metabolite_patient_effects.csv", index=False)

    patient_long = (
        patient.reset_index()
        .melt(
            id_vars=["Patient", "condition"],
            value_vars=selected_features,
            var_name="feature",
            value_name="patient_mean_log_intensity",
        )
        .merge(
            target_inventory[["canonical_metabolite", "observed_feature"]],
            left_on="feature",
            right_on="observed_feature",
            how="left",
        )
    )
    patient_long.to_csv(output_dir / "target_metabolite_patient_long.csv", index=False)

    nec_metadata = sample_metadata.loc[sample_metadata["Group"].eq("nec")].copy()
    nec_columns = [column for column in nec_metadata.index if column in values.index]
    nec_log = np.log1p(values.loc[nec_columns])
    nec_log["Patient"] = nec_metadata.loc[nec_columns, "Patient"].to_numpy()
    nec_log["phase"] = (
        nec_metadata.loc[nec_columns, "Group2"]
        .astype(str)
        .map({"NEC_sick": "sick", "NEC_healthy": "healthy"})
        .to_numpy()
    )
    paired = nec_log.dropna(subset=["phase"]).groupby(
        ["Patient", "phase"],
        as_index=False,
    ).mean(numeric_only=True)
    paired_rows = []
    for feature in selected_features:
        pivot = paired.pivot(index="Patient", columns="phase", values=feature).dropna(
            subset=["healthy", "sick"]
        )
        delta = pivot["sick"] - pivot["healthy"]
        result = paired_sign_flip(delta)
        paired_rows.append(
            {
                "feature": feature,
                "n_paired_nec_patients": len(pivot),
                **result,
            }
        )
    paired_effects = pd.DataFrame(paired_rows)
    paired_effects["fdr_bh"] = bh_fdr(paired_effects["p_value"])
    paired_effects.to_csv(
        output_dir / "target_metabolite_nec_sick_vs_healthy_paired.csv",
        index=False,
    )
    return effects, targeted_effects, patient_long, patient.reset_index()


def canonical_taxon_name(value: str) -> str:
    cleaned = re.sub(r"\|dup\d+$", "", str(value))
    parts = [part.strip() for part in cleaned.split(";")]
    genus = next(
        (
            part[3:].strip().replace("_", " ")
            for part in reversed(parts)
            if part.startswith("g__") and part[3:].strip()
        ),
        "",
    )
    species = next(
        (
            part[3:].strip().replace("_", " ")
            for part in reversed(parts)
            if part.startswith("s__") and part[3:].strip()
        ),
        "",
    )
    if species:
        if genus and not species.lower().startswith(genus.lower()):
            return f"{genus} {species}"
        return species
    for prefix in ["s__", "g__", "f__", "o__", "c__", "p__"]:
        for part in reversed(parts):
            if part.startswith(prefix) and part[len(prefix) :].strip():
                name = part[len(prefix) :].strip()
                return name.replace("_", " ")
    return cleaned


def condition_adjusted_taxon_metabolite_correlations(
    snapshot_results: Path,
    output_dir: Path,
) -> pd.DataFrame:
    directory = snapshot_results / "microbiome_metabolome"
    metadata = pd.read_csv(directory / "infant_metadata_harmonized.csv")
    metadata["sample_key"] = metadata["sample"].astype(str)
    metadata = metadata.set_index("sample_key")
    taxon = pd.read_csv(directory / "taxon_abundance.csv")
    taxon = taxon.rename(columns={taxon.columns[0]: "feature"})
    taxon["canonical_taxon"] = taxon["feature"].map(canonical_taxon_name)
    taxon_sample_columns = [
        column for column in taxon.columns if column not in {"feature", "canonical_taxon"}
    ]
    taxon_numeric = taxon.groupby("canonical_taxon")[taxon_sample_columns].sum()
    metabolite = pd.read_csv(directory / "metabolite_matrix.csv")
    metabolite = metabolite.rename(columns={metabolite.columns[0]: "feature"})
    exact_map = {
        normalize_metabolite_name(feature): feature
        for feature in metabolite["feature"].astype(str)
    }
    selected_metabolites = [
        exact_map[normalize_metabolite_name(name)]
        for name in TARGET_METABOLITES.values()
        if normalize_metabolite_name(name) in exact_map
    ]
    common_samples = sorted(
        set(taxon_sample_columns)
        & set(metabolite.columns[1:])
        & set(metadata.index),
        key=lambda value: int(value) if str(value).isdigit() else str(value),
    )
    eligible_metadata = metadata.loc[common_samples].copy()
    eligible = (
        eligible_metadata["Group"].astype(str).isin(["control", "nec"])
        & ~eligible_metadata["Group"].astype(str).eq("GI")
    )
    common_samples = eligible_metadata.index[eligible].tolist()
    eligible_metadata = eligible_metadata.loc[common_samples]

    counts = taxon_numeric[common_samples].T.astype(float)
    relative = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    taxon_clr_values, _ = clr_transform(relative.to_numpy())
    taxon_clr = pd.DataFrame(
        taxon_clr_values,
        index=relative.index,
        columns=relative.columns,
    )
    taxon_clr["Patient"] = eligible_metadata["Patient"].to_numpy()
    taxon_clr["condition"] = eligible_metadata["Group"].astype(str).to_numpy()
    patient_taxon = taxon_clr.groupby(["Patient", "condition"], as_index=False).mean(
        numeric_only=True
    )

    metabolite_values = (
        metabolite.set_index("feature")[common_samples]
        .T[selected_metabolites]
        .astype(float)
    )
    metabolite_values = np.log1p(metabolite_values)
    metabolite_values["Patient"] = eligible_metadata["Patient"].to_numpy()
    metabolite_values["condition"] = eligible_metadata["Group"].astype(str).to_numpy()
    patient_metabolite = metabolite_values.groupby(
        ["Patient", "condition"],
        as_index=False,
    ).mean(numeric_only=True)
    merged = patient_taxon.merge(
        patient_metabolite,
        on=["Patient", "condition"],
        how="inner",
        suffixes=("_taxon", "_metabolite"),
    )
    taxon_columns = [
        column
        for column in patient_taxon.columns
        if column not in {"Patient", "condition"}
    ]
    variance = merged[taxon_columns].var(axis=0)
    top_taxa = variance.sort_values(ascending=False).head(20).index.tolist()
    condition_numeric = merged["condition"].eq("nec").astype(float).to_numpy()
    design = np.column_stack([np.ones(len(merged)), condition_numeric])
    condition_groups = [
        np.flatnonzero(merged["condition"].to_numpy() == condition)
        for condition in sorted(merged["condition"].unique())
    ]
    n_permutations = 9999
    rng = np.random.default_rng(RANDOM_SEED + 8000)
    permutation_indices = np.tile(
        np.arange(len(merged), dtype=int),
        (n_permutations, 1),
    )
    for group_indices in condition_groups:
        random_order = np.argsort(
            rng.random((n_permutations, len(group_indices))),
            axis=1,
        )
        permutation_indices[:, group_indices] = group_indices[random_order]

    metabolite_residuals: dict[str, np.ndarray] = {}
    permuted_metabolite_residuals: dict[str, np.ndarray] = {}
    for metabolite_name in selected_metabolites:
        metabolite_rank = stats.rankdata(merged[metabolite_name].to_numpy(float))
        residual = metabolite_rank - design @ np.linalg.lstsq(
            design,
            metabolite_rank,
            rcond=None,
        )[0]
        residual = residual - residual.mean()
        norm = float(np.linalg.norm(residual))
        standardized = residual / norm if norm > 0 else np.zeros_like(residual)
        metabolite_residuals[metabolite_name] = standardized
        permuted_metabolite_residuals[metabolite_name] = standardized[
            permutation_indices
        ]

    rows = []
    for taxon_name in top_taxa:
        taxon_rank = stats.rankdata(merged[taxon_name].to_numpy(float))
        taxon_residual = taxon_rank - design @ np.linalg.lstsq(
            design,
            taxon_rank,
            rcond=None,
        )[0]
        taxon_residual = taxon_residual - taxon_residual.mean()
        taxon_norm = float(np.linalg.norm(taxon_residual))
        standardized_taxon = (
            taxon_residual / taxon_norm
            if taxon_norm > 0
            else np.zeros_like(taxon_residual)
        )
        for metabolite_name in selected_metabolites:
            metabolite_residual = metabolite_residuals[metabolite_name]
            rho = float(standardized_taxon @ metabolite_residual)
            if not np.isfinite(rho):
                p_value = float("nan")
            else:
                permuted_rho = (
                    permuted_metabolite_residuals[metabolite_name]
                    @ standardized_taxon
                )
                exceed = int(
                    (np.abs(permuted_rho) >= abs(rho) - 1e-12).sum()
                )
                p_value = float((exceed + 1) / (n_permutations + 1))
            rows.append(
                {
                    "taxon": taxon_name,
                    "metabolite": metabolite_name,
                    "partial_spearman_rho": rho,
                    "p_value": p_value,
                    "n_patients": len(merged),
                    "n_permutations": n_permutations,
                    "adjustment": (
                        "rank residualization on NEC/control condition with "
                        "condition-stratified residual permutation"
                    ),
                }
            )
    frame = pd.DataFrame(rows)
    frame["fdr_bh"] = bh_fdr(frame["p_value"])
    frame.to_csv(
        output_dir / "wandro_patient_level_taxon_metabolite_correlations.csv",
        index=False,
    )
    merged.to_csv(
        output_dir / "wandro_patient_level_multimodal_matrix.csv",
        index=False,
    )
    return frame


def build_sra_model_matrix(
    species_clr: pd.DataFrame,
    pathway_log_cpm: pd.DataFrame,
    snapshot_results: Path,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    species_features = species_clr.add_prefix("species::")
    plausible_columns = [
        column for column in pathway_log_cpm if plausible_microbial_pathway(column)
    ]
    pathway_features = pathway_log_cpm[plausible_columns].add_prefix("pathway::")
    matrix = species_features.join(pathway_features, how="inner")
    labels = pd.read_csv(
        snapshot_results
        / "microbiome_strict"
        / "metaphlan_native_subset"
        / "selected_runs.csv"
    ).set_index("run")
    labels["infant_id"] = labels["isolate"].map(derive_infant_id)
    run_matrix = matrix.copy()
    run_matrix.insert(
        0,
        "condition",
        labels.loc[run_matrix.index, "condition"].astype(str),
    )
    run_matrix.insert(
        0,
        "infant_id",
        labels.loc[run_matrix.index, "infant_id"].astype(str),
    )
    run_matrix.insert(
        0,
        "isolate",
        labels.loc[run_matrix.index, "isolate"].astype(str),
    )
    run_matrix.insert(0, "run", run_matrix.index)
    run_matrix.to_csv(
        output_dir / "sra_run_model_feature_matrix.csv.gz",
        index=False,
        compression="gzip",
    )

    feature_columns = [
        column
        for column in run_matrix.columns
        if column not in {"run", "isolate", "infant_id", "condition"}
    ]
    infant_matrix = (
        run_matrix.groupby(["infant_id", "condition"], as_index=False)[
            feature_columns
        ]
        .mean()
        .sort_values(["condition", "infant_id"])
    )
    infant_matrix.to_csv(
        output_dir / "sra_infant_model_feature_matrix.csv.gz",
        index=False,
        compression="gzip",
    )
    infant_matrix.to_csv(
        output_dir / "sra_model_feature_matrix.csv.gz",
        index=False,
        compression="gzip",
    )
    return run_matrix, infant_matrix


def build_manifest(output_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(output_dir.rglob("*")):
        if path.is_file() and path.name != "microbiome_metabolome_manifest.csv":
            rows.append(
                {
                    "relative_path": path.relative_to(output_dir).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "microbiome_metabolome_manifest.csv", index=False)
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
        "software": {"python": sys.version},
    }
    write_json(output_dir / "microbiome_metabolome_statistics_status.json", status)

    species_effects, species_long, species_clr, species_pca = (
        microbiome_species_analysis(snapshot_results, output_dir)
    )
    pathway_effects, pathway_selected, pathway_log_cpm, pathway_long = (
        humann_pathway_analysis(snapshot_results, output_dir)
    )
    metabolite_effects, targeted_metabolites, patient_metabolites, patient_matrix = (
        metabolome_patient_analysis(snapshot_results, output_dir)
    )
    correlations = condition_adjusted_taxon_metabolite_correlations(
        snapshot_results,
        output_dir,
    )
    run_model_matrix, infant_model_matrix = build_sra_model_matrix(
        species_clr,
        pathway_log_cpm,
        snapshot_results,
        output_dir,
    )

    status.update(
        {
            "status": "completed",
            "completed_utc": utc_now(),
            "metaphlan": {
                "runs": int(species_long["run"].nunique()),
                "infants": int(species_long["infant_id"].nunique()),
                "species": int(species_effects["species"].nunique()),
                "fdr_lt_0_05": int(species_effects["fdr_bh"].lt(0.05).sum()),
                "input_depth": "200,000-spot bounded subset per run",
                "inferential_unit": "infant mean across available runs",
            },
            "humann": {
                "runs": int(pathway_long["run"].nunique()),
                "infants": int(pathway_long["infant_id"].nunique()),
                "pathways_tested": int(len(pathway_effects)),
                "selected_pathways": int(len(pathway_selected)),
                "fdr_lt_0_05": int(pathway_effects["fdr_bh"].lt(0.05).sum()),
                "normalization": "CPM over mapped unstratified pathways followed by log1p",
                "inferential_unit": "infant mean across available runs",
            },
            "metabolome": {
                "features_tested": int(len(metabolite_effects)),
                "targeted_present": int(targeted_metabolites["present"].sum()),
                "nec_patients": int(
                    patient_matrix["condition"].astype(str).eq("nec").sum()
                ),
                "control_patients": int(
                    patient_matrix["condition"].astype(str).eq("control").sum()
                ),
                "fdr_lt_0_05": int(metabolite_effects["fdr_bh"].lt(0.05).sum()),
                "excluded_group": "GI was not treated as a healthy control",
            },
            "patient_level_correlations": int(len(correlations)),
            "sra_model_matrix": {
                "run_rows": int(len(run_model_matrix)),
                "infant_rows": int(len(infant_model_matrix)),
                "features": int(infant_model_matrix.shape[1] - 2),
                "analysis_unit": "infant",
            },
        }
    )
    write_json(output_dir / "microbiome_metabolome_statistics_status.json", status)
    build_manifest(output_dir)
    print(json.dumps(status, ensure_ascii=False, indent=2, default=json_safe))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
