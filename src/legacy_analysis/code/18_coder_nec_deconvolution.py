#!/usr/bin/env python3
"""Run control-donor-regularized cross-platform bulk deconvolution."""

from __future__ import annotations

import itertools
import json
import math
import platform
import shutil
import sqlite3
import subprocess
import sys
import tarfile
from importlib.metadata import version
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy
from scipy import sparse, stats
from scipy.optimize import minimize


SEED = 20260726
MARKERS_PER_STATE = 60
BOOTSTRAPS = 2_000
MIXTURES_PER_DONOR = 100
PSEUDOCOUNT = 1e-4
MIN_FRACTION = 1e-6
RIDGE_GRID = (0.0, 0.01, 0.03, 0.10, 0.30)

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
ATLAS = (
    PROJECT_ROOT
    / "."
    / "04_corrected_results"
    / "01_single_cell_corrected"
    / "corrected_hca_nec_atlas.h5ad"
)
BULK_ROOT = (
    PROJECT_ROOT
    / "."
    / "04_corrected_results"
    / "00_remote_snapshot"
    / "results"
    / "bulk"
)
PUBLIC_ROOT = ROOT / "data" / "public" / "coder_nec"
RESULT_ROOT = ROOT / "results" / "coder_nec"
MANIFESTS = ROOT / "manifests"
PUBLIC_ROOT.mkdir(parents=True, exist_ok=True)
RESULT_ROOT.mkdir(parents=True, exist_ok=True)
MANIFESTS.mkdir(parents=True, exist_ok=True)

ANNOTATION_URL = (
    "https://bioconductor.posit.co/packages/3.23/data/annotation/src/contrib/"
    "hugene10sttranscriptcluster.db_8.8.0.tar.gz"
)
ANNOTATION_ARCHIVE = (
    PUBLIC_ROOT / "hugene10sttranscriptcluster.db_8.8.0.tar.gz"
)
ANNOTATION_SQLITE = PUBLIC_ROOT / "hugene10sttranscriptcluster.sqlite"
GENE_INFO_URL = (
    "https://ftp.ncbi.nlm.nih.gov/gene/DATA/GENE_INFO/Mammalia/"
    "Homo_sapiens.gene_info.gz"
)
GENE_INFO_PATH = PUBLIC_ROOT / "Homo_sapiens.gene_info.gz"

STATES = [
    "M_cell",
    "LEC",
    "T_reg",
    "T_NK",
    "epithelial",
    "myeloid",
    "vascular_endothelial",
    "fibroblast",
]

CONTRASTS = {
    "GSE46619": [
        ("nec_vs_control", "nec", "control"),
        ("nec_vs_sip", "nec", "sip"),
    ],
    "GSE64801": [("nec_vs_control", "nec", "control")],
}


def ensure_local(path: Path) -> Path:
    resolved = path.resolve()
    root = ROOT.resolve()
    if resolved != root and root not in resolved.parents:
        raise RuntimeError(f"Refusing write outside local analysis root: {resolved}")
    return resolved


def bh_adjust(values: pd.Series | np.ndarray) -> np.ndarray:
    raw = np.asarray(values, dtype=float)
    adjusted = np.full(raw.shape, np.nan, dtype=float)
    finite = np.isfinite(raw)
    if not finite.any():
        return adjusted
    p = raw[finite]
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    restored = np.empty_like(q)
    restored[order] = np.minimum(q, 1.0)
    adjusted[finite] = restored
    return adjusted


def hedges_g(case: np.ndarray, control: np.ndarray) -> float:
    n_case = len(case)
    n_control = len(control)
    denominator = n_case + n_control - 2
    if denominator <= 0:
        return np.nan
    pooled_variance = (
        (n_case - 1) * np.var(case, ddof=1)
        + (n_control - 1) * np.var(control, ddof=1)
    ) / denominator
    if pooled_variance <= 0:
        return np.nan
    correction = 1 - 3 / (4 * (n_case + n_control) - 9)
    return float(
        correction
        * (np.mean(case) - np.mean(control))
        / math.sqrt(pooled_variance)
    )


def exact_permutation(
    values: np.ndarray,
    labels: np.ndarray,
) -> tuple[float, float, int]:
    case_count = int(labels.sum())
    combinations = list(
        itertools.combinations(range(len(labels)), case_count)
    )
    assignment = np.zeros((len(combinations), len(labels)), dtype=float)
    for row, indices in enumerate(combinations):
        assignment[row, list(indices)] = 1.0
    null = (
        assignment @ values / case_count
        - (1 - assignment) @ values / (len(labels) - case_count)
    )
    observed = float(
        np.mean(values[labels == 1]) - np.mean(values[labels == 0])
    )
    pvalue = float(np.mean(np.abs(null) >= abs(observed) - 1e-12))
    return observed, pvalue, len(combinations)


def bootstrap_interval(
    case: np.ndarray,
    control: np.ndarray,
    offset: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(SEED + offset)
    case_draw = rng.choice(case, size=(BOOTSTRAPS, len(case)), replace=True)
    control_draw = rng.choice(
        control, size=(BOOTSTRAPS, len(control)), replace=True
    )
    delta = case_draw.mean(axis=1) - control_draw.mean(axis=1)
    return tuple(np.quantile(delta, [0.025, 0.975]).astype(float))


def download_annotation() -> None:
    if (
        ANNOTATION_ARCHIVE.exists()
        and ANNOTATION_ARCHIVE.stat().st_size == 3_866_814
    ):
        return
    curl = shutil.which("curl.exe") or shutil.which("curl")
    if not curl:
        raise RuntimeError("curl is required for GPL6244 annotation")
    temporary = ensure_local(Path(str(ANNOTATION_ARCHIVE) + ".part"))
    temporary.unlink(missing_ok=True)
    completed = subprocess.run(
        [
            curl,
            "--location",
            "--fail",
            "--silent",
            "--show-error",
            "--retry",
            "8",
            "--retry-all-errors",
            "--retry-delay",
            "4",
            "--output",
            str(temporary),
            ANNOTATION_URL,
        ],
        check=False,
    )
    if completed.returncode != 0:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            f"Bioconductor annotation download failed with code "
            f"{completed.returncode}"
        )
    if temporary.stat().st_size != 3_866_814:
        temporary.unlink(missing_ok=True)
        raise RuntimeError("Bioconductor annotation has an unexpected size")
    temporary.replace(ANNOTATION_ARCHIVE)


def download_gene_info() -> None:
    if GENE_INFO_PATH.exists() and GENE_INFO_PATH.stat().st_size > 4_000_000:
        return
    curl = shutil.which("curl.exe") or shutil.which("curl")
    if not curl:
        raise RuntimeError("curl is required for NCBI Gene annotation")
    temporary = ensure_local(Path(str(GENE_INFO_PATH) + ".part"))
    temporary.unlink(missing_ok=True)
    completed = subprocess.run(
        [
            curl,
            "--location",
            "--fail",
            "--silent",
            "--show-error",
            "--retry",
            "8",
            "--retry-all-errors",
            "--retry-delay",
            "4",
            "--output",
            str(temporary),
            GENE_INFO_URL,
        ],
        check=False,
    )
    if completed.returncode != 0:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            f"NCBI Gene annotation download failed with code "
            f"{completed.returncode}"
        )
    if temporary.stat().st_size <= 4_000_000:
        temporary.unlink(missing_ok=True)
        raise RuntimeError("NCBI Gene annotation is unexpectedly small")
    temporary.replace(GENE_INFO_PATH)


def parse_annotation() -> pd.DataFrame:
    if not ANNOTATION_SQLITE.exists():
        with tarfile.open(ANNOTATION_ARCHIVE, mode="r:gz") as archive:
            members = [
                member
                for member in archive.getmembers()
                if member.isfile() and member.name.endswith(".sqlite")
            ]
            if len(members) != 1:
                raise RuntimeError(
                    f"Expected one ChipDb SQLite file, found {len(members)}"
                )
            handle = archive.extractfile(members[0])
            if handle is None:
                raise RuntimeError("Cannot extract ChipDb SQLite file")
            ANNOTATION_SQLITE.write_bytes(handle.read())
    connection = sqlite3.connect(ANNOTATION_SQLITE)
    try:
        table_names = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        if "probes" not in table_names:
            raise RuntimeError(
                "ChipDb schema lacks a probes table: "
                + ", ".join(sorted(table_names))
            )
        probes = pd.read_sql_query(
            """
            SELECT CAST(probes.probe_id AS TEXT) AS probe_id,
                   CAST(probes.gene_id AS TEXT) AS gene_id,
                   probes.is_multiple AS is_multiple
            FROM probes
            WHERE probes.gene_id IS NOT NULL
            """,
            connection,
        )
    finally:
        connection.close()
    gene_info = pd.read_csv(
        GENE_INFO_PATH,
        sep="\t",
        compression="gzip",
        dtype=str,
        usecols=["GeneID", "Symbol"],
        low_memory=False,
    ).rename(columns={"GeneID": "gene_id", "Symbol": "gene"})
    mapping = probes.merge(gene_info, on="gene_id", how="inner")
    mapping = mapping[mapping["is_multiple"].astype(int).eq(0)].copy()
    mapping["probe_id"] = mapping["probe_id"].astype(str)
    mapping["gene"] = mapping["gene"].astype(str).str.strip()
    mapping = mapping[
        ~mapping["gene"].isin(["", "---", "nan", "NA"])
    ].drop_duplicates("probe_id")
    mapping.to_csv(
        ensure_local(PUBLIC_ROOT / "GPL6244_probe_to_gene.csv"),
        index=False,
    )
    return mapping


def donor_state_reference() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    backed = ad.read_h5ad(ATLAS, backed="r")
    mask = (
        backed.obs["condition"].astype(str).eq("control")
        & backed.obs["selection"].astype(str).eq("unselected")
    ).to_numpy()
    reference = backed[mask].to_memory()
    obs = reference.obs.copy()
    obs["cell_state"] = obs["cell_state"].astype(str).replace(
        {"epithelial_stem": "epithelial"}
    )
    matrix = reference.X
    if not sparse.issparse(matrix):
        matrix = sparse.csr_matrix(matrix)
    else:
        matrix = matrix.tocsr()
    genes = reference.var_names.astype(str).to_numpy()
    rows: list[pd.DataFrame] = []
    count_rows: list[dict[str, object]] = []
    for donor in sorted(obs["donor_id"].astype(str).unique()):
        for state in STATES:
            positions = np.flatnonzero(
                obs["donor_id"].astype(str).eq(donor).to_numpy()
                & obs["cell_state"].eq(state).to_numpy()
            )
            if len(positions) < 5:
                continue
            mean_expression = np.asarray(
                matrix[positions].mean(axis=0)
            ).ravel()
            rows.append(
                pd.DataFrame(
                    {
                        "donor_id": donor,
                        "cell_state": state,
                        "gene": genes,
                        "mean_log_expression": mean_expression,
                    }
                )
            )
            count_rows.append(
                {
                    "donor_id": donor,
                    "cell_state": state,
                    "cells": len(positions),
                }
            )
    donor_state = pd.concat(rows, ignore_index=True)
    counts = pd.DataFrame(count_rows)
    if counts["donor_id"].nunique() != 5:
        raise RuntimeError("Expected five unselected control donors")
    signature = (
        donor_state.groupby(["gene", "cell_state"], observed=True)[
            "mean_log_expression"
        ]
        .median()
        .unstack("cell_state")
        .reindex(columns=STATES)
    )
    signature.to_csv(
        ensure_local(RESULT_ROOT / "coder_control_donor_signature_all_genes.csv")
    )
    counts.to_csv(
        ensure_local(RESULT_ROOT / "coder_reference_cell_counts.csv"),
        index=False,
    )
    return donor_state, signature, counts


def select_markers(signature: pd.DataFrame) -> pd.DataFrame:
    values = signature.to_numpy(dtype=float)
    order = np.argsort(values, axis=1)
    top = values[np.arange(len(values)), order[:, -1]]
    second = values[np.arange(len(values)), order[:, -2]]
    dominant = np.asarray(STATES, dtype=object)[order[:, -1]]
    marker = pd.DataFrame(
        {
            "gene": signature.index.astype(str),
            "assigned_state": dominant,
            "state_mean_expression": top,
            "specificity_margin": top - second,
        }
    )
    marker = marker[
        (marker["state_mean_expression"] >= 0.10)
        & (marker["specificity_margin"] > 0)
        & ~marker["gene"].str.match(r"^(MT-|RPL|RPS)")
    ].copy()
    selected = (
        marker.sort_values(
            ["assigned_state", "specificity_margin"],
            ascending=[True, False],
        )
        .groupby("assigned_state", observed=True)
        .head(MARKERS_PER_STATE)
    )
    missing = sorted(set(STATES) - set(selected["assigned_state"]))
    if missing:
        raise RuntimeError(f"No markers selected for states: {missing}")
    selected.to_csv(
        ensure_local(RESULT_ROOT / "coder_reference_markers.csv"),
        index=False,
    )
    return selected


def aggregate_bulk(
    accession: str,
    mapping: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    expression_path = BULK_ROOT / accession / "expression_gene_matrix.csv"
    metadata_path = BULK_ROOT / accession / "metadata_harmonized.csv"
    expression = pd.read_csv(expression_path, index_col=0)
    expression.index = expression.index.astype(str)
    expression.columns = expression.columns.astype(str)
    audit_rows: list[dict[str, object]] = []
    if accession == "GSE46619":
        map_series = mapping.set_index("probe_id")["gene"]
        feature = expression.index.to_series()
        numeric = feature.str.fullmatch(r"\d+")
        probe_gene = feature.where(~numeric, feature.map(map_series))
        audit_rows.append(
            {
                "accession": accession,
                "input_features": len(expression),
                "mapped_features": int(probe_gene.notna().sum()),
                "mapped_fraction": float(probe_gene.notna().mean()),
                "preannotated_gene_symbols": int((~numeric).sum()),
                "numeric_probe_ids": int(numeric.sum()),
                "numeric_probe_ids_rescued": int(
                    probe_gene[numeric].notna().sum()
                ),
            }
        )
        mapped = expression.loc[probe_gene.notna()].copy()
        mapped["gene"] = probe_gene.loc[probe_gene.notna()].to_numpy()
        expression = mapped.groupby("gene", observed=True).median()
    else:
        expression.index = expression.index.str.split(".", n=1).str[0]
        expression = expression.groupby(level=0).median()
        audit_rows.append(
            {
                "accession": accession,
                "input_features": len(expression),
                "mapped_features": len(expression),
                "mapped_fraction": 1.0,
            }
        )
    expression.index = expression.index.astype(str)
    metadata = pd.read_csv(metadata_path, low_memory=False)
    metadata["geo_accession"] = metadata["geo_accession"].astype(str)
    metadata = metadata.set_index("geo_accession").loc[expression.columns]
    audit = pd.DataFrame(audit_rows)
    expression.to_csv(
        ensure_local(RESULT_ROOT / f"coder_{accession}_gene_expression.csv")
    )
    return expression, metadata, audit


def make_signature(
    donor_state: pd.DataFrame,
    exclude_donor: str | None,
) -> pd.DataFrame:
    frame = donor_state
    if exclude_donor is not None:
        frame = frame[frame["donor_id"] != exclude_donor]
    return (
        frame.groupby(["gene", "cell_state"], observed=True)[
            "mean_log_expression"
        ]
        .median()
        .unstack("cell_state")
        .reindex(columns=STATES)
    )


def rank_z(frame: pd.DataFrame) -> np.ndarray:
    ranks = frame.rank(axis=0, method="average", pct=True)
    values = ranks.to_numpy(dtype=float)
    mean = values.mean(axis=0, keepdims=True)
    standard_deviation = values.std(axis=0, ddof=1, keepdims=True)
    standard_deviation[standard_deviation <= 0] = 1.0
    return (values - mean) / standard_deviation


def project(
    signature: pd.DataFrame,
    mixture: pd.DataFrame,
    marker_genes: list[str],
    ridge: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    common = [
        gene
        for gene in marker_genes
        if gene in signature.index and gene in mixture.index
    ]
    if len(common) < 100:
        raise RuntimeError(
            f"Only {len(common)} reference markers overlap the mixture"
        )
    basis = rank_z(signature.loc[common, STATES])
    observations = rank_z(mixture.loc[common])
    result_rows: list[dict[str, object]] = []
    fit_rows: list[dict[str, object]] = []
    prior = np.repeat(1 / len(STATES), len(STATES))
    bounds = [(MIN_FRACTION, 1.0) for _ in STATES]
    constraint = {
        "type": "eq",
        "fun": lambda coefficient: coefficient.sum() - 1.0,
        "jac": lambda coefficient: np.ones_like(coefficient),
    }
    for sample_index, sample in enumerate(mixture.columns.astype(str)):
        target = observations[:, sample_index]

        def objective(coefficient: np.ndarray) -> float:
            residual = basis @ coefficient - target
            return float(
                np.mean(residual**2)
                + ridge * np.sum((coefficient - prior) ** 2)
            )

        def gradient(coefficient: np.ndarray) -> np.ndarray:
            residual = basis @ coefficient - target
            return (
                2 * basis.T @ residual / len(target)
                + 2 * ridge * (coefficient - prior)
            )

        optimized = minimize(
            objective,
            prior,
            jac=gradient,
            method="SLSQP",
            bounds=bounds,
            constraints=[constraint],
            options={"ftol": 1e-10, "maxiter": 500},
        )
        if not optimized.success:
            raise RuntimeError(
                f"Simplex projection failed for {sample}: {optimized.message}"
            )
        coefficient = np.clip(
            optimized.x.astype(float), MIN_FRACTION, 1.0
        )
        coefficient /= coefficient.sum()
        reconstructed = basis @ coefficient
        correlation = stats.pearsonr(target, reconstructed).statistic
        for state, value in zip(STATES, coefficient):
            result_rows.append(
                {
                    "sample": sample,
                    "cell_state": state,
                    "relative_fraction": float(value),
                }
            )
        fit_rows.append(
            {
                "sample": sample,
                "marker_genes": len(common),
                "reconstruction_correlation": float(correlation),
                "residual_sum_squares": float(
                    np.sum((target - reconstructed) ** 2)
                ),
                "ridge": ridge,
            }
        )
    return pd.DataFrame(result_rows), pd.DataFrame(fit_rows)


def deconvolve_accession(
    accession: str,
    expression: pd.DataFrame,
    metadata: pd.DataFrame,
    donor_state: pd.DataFrame,
    marker_genes: list[str],
    ridge: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    control_donors = sorted(donor_state["donor_id"].astype(str).unique())
    iterations: list[pd.DataFrame] = []
    fits: list[pd.DataFrame] = []
    for iteration, excluded in enumerate([None] + control_donors):
        signature = make_signature(donor_state, excluded)
        projected, fit = project(
            signature, expression, marker_genes, ridge
        )
        label = "all_control_donors" if excluded is None else f"leave_out_{excluded}"
        projected["reference_iteration"] = label
        fit["reference_iteration"] = label
        iterations.append(projected)
        fits.append(fit)
    raw = pd.concat(iterations, ignore_index=True)
    summary = (
        raw.groupby(["sample", "cell_state"], observed=True)[
            "relative_fraction"
        ]
        .agg(
            relative_fraction="median",
            uncertainty_q25=lambda value: value.quantile(0.25),
            uncertainty_q75=lambda value: value.quantile(0.75),
            uncertainty_range=lambda value: value.max() - value.min(),
        )
        .reset_index()
    )
    summary["accession"] = accession
    summary["condition"] = summary["sample"].map(metadata["condition"])
    fit_summary = (
        pd.concat(fits, ignore_index=True)
        .groupby("sample", observed=True)
        .agg(
            marker_genes=("marker_genes", "min"),
            median_reconstruction_correlation=(
                "reconstruction_correlation",
                "median",
            ),
            minimum_reconstruction_correlation=(
                "reconstruction_correlation",
                "min",
            ),
            median_residual_sum_squares=(
                "residual_sum_squares",
                "median",
            ),
        )
        .reset_index()
    )
    fit_summary["accession"] = accession
    raw["accession"] = accession
    raw.to_csv(
        ensure_local(
            RESULT_ROOT / f"coder_{accession}_reference_ensemble.csv"
        ),
        index=False,
    )
    return summary, fit_summary


def benchmark(
    donor_state: pd.DataFrame,
    marker_genes: list[str],
    ridge: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(SEED)
    rows: list[pd.DataFrame] = []
    donors = sorted(donor_state["donor_id"].astype(str).unique())
    for donor_index, donor in enumerate(donors):
        held = (
            donor_state[donor_state["donor_id"] == donor]
            .pivot(index="gene", columns="cell_state", values="mean_log_expression")
            .reindex(columns=STATES)
        )
        training = make_signature(donor_state, donor)
        available_states = [
            state for state in STATES if held[state].notna().all()
        ]
        if len(available_states) < 6:
            raise RuntimeError(
                f"Held-out donor {donor} has too few reference states"
            )
        common = [
            gene
            for gene in marker_genes
            if gene in held.index
            and gene in training.index
            and np.isfinite(held.loc[gene, available_states]).all()
        ]
        held_linear = np.expm1(
            held.loc[common, available_states].clip(lower=0)
        )
        weights = rng.dirichlet(
            np.repeat(1.0, len(available_states)),
            size=MIXTURES_PER_DONOR,
        )
        mixed_linear = held_linear.to_numpy(dtype=float) @ weights.T
        mixtures = pd.DataFrame(
            np.log1p(mixed_linear),
            index=common,
            columns=[
                f"{donor}_mixture_{value:03d}"
                for value in range(MIXTURES_PER_DONOR)
            ],
        )
        inferred, _ = project(training, mixtures, common, ridge)
        inferred = inferred.pivot(
            index="sample",
            columns="cell_state",
            values="relative_fraction",
        ).reindex(columns=STATES)
        true = pd.DataFrame(
            0.0, index=mixtures.columns, columns=STATES
        )
        true.loc[:, available_states] = weights
        true.index.name = "sample"
        true.columns.name = "cell_state"
        long = (
            inferred.stack(future_stack=True)
            .rename("inferred_fraction")
            .reset_index()
            .merge(
                true.stack(future_stack=True)
                .rename("true_fraction")
                .reset_index(),
                on=["sample", "cell_state"],
            )
        )
        long["held_out_donor"] = donor
        long["ridge"] = ridge
        rows.append(long)
    benchmark_long = pd.concat(rows, ignore_index=True)
    metrics: list[dict[str, object]] = []
    for state, frame in benchmark_long.groupby("cell_state", observed=True):
        metrics.append(
            {
                "cell_state": state,
                "ridge": ridge,
                "spearman_rho": float(
                    stats.spearmanr(
                        frame["true_fraction"],
                        frame["inferred_fraction"],
                    ).statistic
                ),
                "mae": float(
                    np.mean(
                        np.abs(
                            frame["true_fraction"]
                            - frame["inferred_fraction"]
                        )
                    )
                ),
                "rmse": float(
                    np.sqrt(
                        np.mean(
                            (
                                frame["true_fraction"]
                                - frame["inferred_fraction"]
                            )
                            ** 2
                        )
                    )
                ),
            }
        )
    metric_frame = pd.DataFrame(metrics)
    return benchmark_long, metric_frame


def tune_ridge(
    donor_state: pd.DataFrame,
    marker_genes: list[str],
) -> tuple[float, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    long_tables: list[pd.DataFrame] = []
    metric_tables: list[pd.DataFrame] = []
    score_rows: list[dict[str, float]] = []
    for ridge in RIDGE_GRID:
        benchmark_long, metrics = benchmark(
            donor_state, marker_genes, ridge
        )
        long_tables.append(benchmark_long)
        metric_tables.append(metrics)
        score_rows.append(
            {
                "ridge": ridge,
                "overall_mae": float(
                    np.mean(
                        np.abs(
                            benchmark_long["true_fraction"]
                            - benchmark_long["inferred_fraction"]
                        )
                    )
                ),
                "median_state_mae": float(metrics["mae"].median()),
                "maximum_state_mae": float(metrics["mae"].max()),
                "median_state_spearman": float(
                    metrics["spearman_rho"].median()
                ),
            }
        )
    scores = pd.DataFrame(score_rows)
    scores["selection_objective"] = (
        scores["overall_mae"] + 0.25 * scores["maximum_state_mae"]
    )
    scores.sort_values(
        ["selection_objective", "ridge"], inplace=True
    )
    selected_ridge = float(scores.iloc[0]["ridge"])
    all_long = pd.concat(long_tables, ignore_index=True)
    all_metrics = pd.concat(metric_tables, ignore_index=True)
    selected_long = all_long[all_long["ridge"] == selected_ridge].copy()
    selected_metrics = all_metrics[
        all_metrics["ridge"] == selected_ridge
    ].copy()
    scores.to_csv(
        ensure_local(RESULT_ROOT / "coder_ridge_selection.csv"),
        index=False,
    )
    selected_long.to_csv(
        ensure_local(RESULT_ROOT / "coder_held_out_donor_mixtures.csv"),
        index=False,
    )
    selected_metrics.to_csv(
        ensure_local(RESULT_ROOT / "coder_benchmark_metrics.csv"),
        index=False,
    )
    return selected_ridge, selected_long, selected_metrics, scores


def composition_effects(
    deconvolution: pd.DataFrame,
    benchmark_metrics: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    benchmark_lookup = benchmark_metrics.set_index("cell_state")
    for accession, contrasts in CONTRASTS.items():
        accession_frame = deconvolution[
            deconvolution["accession"] == accession
        ]
        detected_fraction = (
            accession_frame.assign(
                detected=accession_frame["relative_fraction"]
                > 10 * MIN_FRACTION
            )
            .groupby("cell_state", observed=True)["detected"]
            .mean()
        )
        wide = accession_frame.pivot(
            index=["sample", "condition"],
            columns="cell_state",
            values="relative_fraction",
        ).reindex(columns=STATES)
        clr = np.log(wide + PSEUDOCOUNT)
        clr = clr.sub(clr.mean(axis=1), axis=0)
        for contrast_index, (contrast, case, control) in enumerate(contrasts):
            mask = wide.index.get_level_values("condition").isin(
                [case, control]
            )
            contrast_clr = clr.loc[mask]
            condition = contrast_clr.index.get_level_values("condition")
            labels = np.asarray(condition == case, dtype=int)
            for state_index, state in enumerate(STATES):
                values = contrast_clr[state].to_numpy(dtype=float)
                effect, pvalue, assignments = exact_permutation(values, labels)
                case_values = values[labels == 1]
                control_values = values[labels == 0]
                ci_low, ci_high = bootstrap_interval(
                    case_values,
                    control_values,
                    contrast_index * 100 + state_index,
                )
                rows.append(
                    {
                        "accession": accession,
                        "contrast": contrast,
                        "case": case,
                        "control": control,
                        "cell_state": state,
                        "case_samples": len(case_values),
                        "control_samples": len(control_values),
                        "case_mean_clr": float(case_values.mean()),
                        "control_mean_clr": float(control_values.mean()),
                        "case_minus_control_clr": effect,
                        "hedges_g": hedges_g(case_values, control_values),
                        "bootstrap_ci_low": ci_low,
                        "bootstrap_ci_high": ci_high,
                        "exact_permutation_pvalue": pvalue,
                        "exact_assignments": assignments,
                        "benchmark_spearman": float(
                            benchmark_lookup.loc[state, "spearman_rho"]
                        ),
                        "benchmark_mae": float(
                            benchmark_lookup.loc[state, "mae"]
                        ),
                        "detected_sample_fraction": float(
                            detected_fraction.loc[state]
                        ),
                    }
                )
    effects = pd.DataFrame(rows)
    effects["interpretation_eligible"] = (
        (effects["benchmark_spearman"] >= 0.70)
        & (effects["benchmark_mae"] <= 0.10)
        & (effects["detected_sample_fraction"] >= 0.50)
    )
    effects["fdr_bh"] = np.nan
    for _, positions in effects.groupby(
        ["accession", "contrast"], observed=True
    ).groups.items():
        effects.loc[positions, "fdr_bh"] = bh_adjust(
            effects.loc[positions, "exact_permutation_pvalue"]
        )
    effects.sort_values(
        ["accession", "contrast", "fdr_bh", "cell_state"],
        inplace=True,
    )
    effects.to_csv(
        ensure_local(RESULT_ROOT / "coder_bulk_composition_effects.csv"),
        index=False,
    )
    return effects


def main() -> None:
    download_annotation()
    download_gene_info()
    mapping = parse_annotation()
    donor_state, signature, reference_counts = donor_state_reference()
    markers = select_markers(signature)
    marker_genes = markers["gene"].astype(str).tolist()
    selected_ridge, benchmark_long, benchmark_metrics, ridge_scores = (
        tune_ridge(donor_state, marker_genes)
    )

    deconvolution_rows: list[pd.DataFrame] = []
    fit_rows: list[pd.DataFrame] = []
    mapping_rows: list[pd.DataFrame] = []
    for accession in CONTRASTS:
        expression, metadata, audit = aggregate_bulk(accession, mapping)
        deconvolution, fit = deconvolve_accession(
            accession,
            expression,
            metadata,
            donor_state,
            marker_genes,
            selected_ridge,
        )
        deconvolution_rows.append(deconvolution)
        fit_rows.append(fit)
        mapping_rows.append(audit)
    deconvolution = pd.concat(deconvolution_rows, ignore_index=True)
    fit = pd.concat(fit_rows, ignore_index=True)
    mapping_audit = pd.concat(mapping_rows, ignore_index=True)
    deconvolution.to_csv(
        ensure_local(RESULT_ROOT / "coder_bulk_relative_fractions.csv"),
        index=False,
    )
    fit.to_csv(
        ensure_local(RESULT_ROOT / "coder_bulk_reconstruction_qc.csv"),
        index=False,
    )
    mapping_audit.to_csv(
        ensure_local(RESULT_ROOT / "coder_bulk_mapping_audit.csv"),
        index=False,
    )

    effects = composition_effects(deconvolution, benchmark_metrics)
    manifest = {
        "status": "PASS",
        "analysis": "CoDeR-NEC",
        "algorithm": (
            "control-donor-regularized rank-space deconvolution; 60 state-specific "
            "markers per state; simplex-constrained quadratic projection; "
            "ridge selection using held-out-donor synthetic mixtures; "
            "leave-one-control-donor reference ensemble; exact sample-label "
            "permutations on CLR relative fractions"
        ),
        "statistical_unit": "bulk tissue sample",
        "reference_donors": int(reference_counts["donor_id"].nunique()),
        "reference_states": STATES,
        "reference_markers": int(len(markers)),
        "selected_ridge": selected_ridge,
        "bulk_datasets": sorted(CONTRASTS),
        "bulk_samples": int(deconvolution["sample"].nunique()),
        "deconvolved_sample_state_rows": int(len(deconvolution)),
        "median_reconstruction_correlation": float(
            fit["median_reconstruction_correlation"].median()
        ),
        "minimum_reconstruction_correlation": float(
            fit["minimum_reconstruction_correlation"].min()
        ),
        "benchmark_mixtures": int(
            benchmark_long["sample"].nunique()
        ),
        "benchmark_median_spearman": float(
            benchmark_metrics["spearman_rho"].median()
        ),
        "benchmark_maximum_mae": float(benchmark_metrics["mae"].max()),
        "composition_tests": int(len(effects)),
        "composition_fdr_0_05_all_states": int(
            (effects["fdr_bh"] < 0.05).sum()
        ),
        "composition_interpretation_eligible_tests": int(
            effects["interpretation_eligible"].sum()
        ),
        "composition_fdr_0_05_interpretation_eligible": int(
            (
                (effects["fdr_bh"] < 0.05)
                & effects["interpretation_eligible"]
            ).sum()
        ),
        "gpl6244_mapped_probes": int(mapping["probe_id"].nunique()),
        "versions": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "anndata": version("anndata"),
        },
        "cloud_files_written": 0,
    }
    (MANIFESTS / "coder_nec_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
