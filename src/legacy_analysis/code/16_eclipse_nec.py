#!/usr/bin/env python3
"""Run ECLIPSE-NEC cross-tissue methylome and single-cell integration."""

from __future__ import annotations

import itertools
import json
import math
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score


SEED = 20260726
BIN_SIZE = 1_000
MIN_COVERAGE = 10
MIN_CPGS = 3
MIN_GROUP_FRACTION = 0.60
EMPIRICAL_WINDOWS = 500
EMPIRICAL_MAX_ASSIGNMENTS = 10_000

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
PUBLIC_ROOT = ROOT / "data" / "public" / "eclipse_nec"
PROCESSED_ROOT = ROOT / "data" / "processed" / "eclipse_nec"
BIN_ROOT = PROCESSED_ROOT / "bin_1kb"
RESULT_ROOT = ROOT / "results" / "eclipse_nec"
MANIFESTS = ROOT / "manifests"
RESULT_ROOT.mkdir(parents=True, exist_ok=True)
MANIFESTS.mkdir(parents=True, exist_ok=True)

COHORTS = {
    "ileum": {
        "tissue": "ileum tissue",
        "case": "nec",
        "comparator": "non_nec",
    },
    "colon": {
        "tissue": "colon tissue",
        "case": "nec",
        "comparator": "non_nec",
    },
    "stool": {
        "tissue": "stool",
        "case": "nec",
        "comparator": "control",
    },
    "blood": {
        "tissue": "blood",
        "case": "nec",
        "comparator": "control",
    },
}

STATE_FILES = {
    "M cell": "M_cell",
    "LEC": "LEC",
    "T/NK": "T_NK",
    "Treg": "T_reg",
}

MODULES = {
    "Host-response circuit": [
        "CXCL16",
        "CXCR6",
        "VEGFC",
        "FLT4",
        "EFNA1",
        "EPHA2",
        "TNF",
        "ICAM1",
        "IL22",
        "IL22RA1",
        "CCL21",
        "CCR7",
        "PROX1",
        "LYVE1",
        "PDPN",
        "ACKR4",
        "GP2",
        "SPIB",
        "SOX8",
        "FOXP3",
        "IL2RA",
        "CTLA4",
        "TIGIT",
        "RORC",
        "AHR",
        "KIT",
    ],
    "Ferroptosis": [
        "GPX4",
        "ACSL4",
        "SLC7A11",
        "FTH1",
        "TFRC",
        "NFE2L2",
        "HMOX1",
        "LPCAT3",
        "AIFM2",
        "ALOX15",
        "PTGS2",
    ],
    "FXR-bile acid": [
        "NR1H4",
        "FGF19",
        "FGFR4",
        "KLB",
        "SLC10A2",
        "SLC51A",
        "SLC51B",
        "CYP7A1",
        "FABP6",
    ],
    "Hypoxia-epithelial injury": [
        "HIF1A",
        "EPAS1",
        "VEGFA",
        "OLFM4",
        "LGR5",
        "DEFA5",
        "DEFA6",
        "IL1B",
        "NLRP3",
    ],
    "M-cell identity": ["SPIB", "GP2", "SOX8", "TNFRSF11A", "CCL20", "MARCKSL1"],
    "LEC identity": ["PROX1", "LYVE1", "PDPN", "FLT4", "CCL21", "ACKR4"],
    "ILC3 identity": ["RORC", "IL7R", "KIT", "AHR", "IL22", "NCR1"],
    "Resident T identity": ["CD69", "ITGAE", "CXCR6", "ZNF683", "RUNX3", "CCL5"],
    "Treg identity": ["FOXP3", "IL2RA", "CTLA4", "TIGIT", "IKZF2", "CCR7"],
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


def acat(values: np.ndarray) -> float:
    p = np.asarray(values, dtype=float)
    p = p[np.isfinite(p)]
    if p.size == 0:
        return np.nan
    p = np.clip(p, 1e-15, 1 - 1e-15)
    statistic = np.mean(np.tan((0.5 - p) * np.pi))
    combined = 0.5 - np.arctan(statistic) / np.pi
    return float(np.clip(combined, 0.0, 1.0))


def spearman_inference(
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[float, float, str]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    x = x[keep]
    y = y[keep]
    if len(x) < 4:
        return np.nan, np.nan, "not estimable"
    observed = float(stats.spearmanr(x, y).statistic)
    if len(x) <= 9:
        x_rank = stats.rankdata(x, method="average")
        y_rank = stats.rankdata(y, method="average")
        x_centered = x_rank - x_rank.mean()
        y_centered = y_rank - y_rank.mean()
        denominator = float(
            np.sqrt(
                np.sum(x_centered**2)
                * np.sum(y_centered**2)
            )
        )
        exceedances = 0
        assignments = 0
        for permutation in itertools.permutations(range(len(y))):
            statistic = float(
                x_centered
                @ y_centered[np.asarray(permutation, dtype=int)]
                / denominator
            )
            exceedances += int(
                abs(statistic) >= abs(observed) - 1e-12
            )
            assignments += 1
        return (
            observed,
            float(exceedances / assignments),
            f"exact {assignments} gene-label permutations",
        )
    result = stats.spearmanr(x, y)
    return (
        observed,
        float(result.pvalue),
        "asymptotic Spearman test",
    )


def hedges_g_matrix(
    case_values: np.ndarray,
    comparator_values: np.ndarray,
) -> np.ndarray:
    n_case = np.isfinite(case_values).sum(axis=1)
    n_comparator = np.isfinite(comparator_values).sum(axis=1)
    mean_case = np.nanmean(case_values, axis=1)
    mean_comparator = np.nanmean(comparator_values, axis=1)
    var_case = np.nanvar(case_values, axis=1, ddof=1)
    var_comparator = np.nanvar(comparator_values, axis=1, ddof=1)
    denominator = n_case + n_comparator - 2
    pooled = np.divide(
        (n_case - 1) * var_case + (n_comparator - 1) * var_comparator,
        denominator,
        out=np.full(len(denominator), np.nan),
        where=denominator > 0,
    )
    effect = np.divide(
        mean_case - mean_comparator,
        np.sqrt(pooled),
        out=np.full(len(pooled), np.nan),
        where=pooled > 0,
    )
    correction = 1 - np.divide(
        3.0,
        4 * (n_case + n_comparator) - 9,
        out=np.zeros(len(n_case), dtype=float),
        where=(4 * (n_case + n_comparator) - 9) > 0,
    )
    return effect * correction


def build_tissue_matrix(
    tissue_metadata: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sample_frames: list[pd.DataFrame] = []
    for row in tissue_metadata.itertuples(index=False):
        path = BIN_ROOT / f"{row.gsm}_bin_1kb.parquet"
        frame = pd.read_parquet(
            path,
            columns=[
                "chromosome",
                "bin_start",
                "coverage",
                "cpgs",
                "beta",
            ],
        )
        frame = frame[
            (frame["coverage"] >= MIN_COVERAGE)
            & (frame["cpgs"] >= MIN_CPGS)
        ].copy()
        if frame.duplicated(["chromosome", "bin_start"]).any():
            raise RuntimeError(f"{row.gsm}: duplicate 1-kb bins")
        indexed = frame.set_index(["chromosome", "bin_start"])[
            ["beta", "coverage", "cpgs"]
        ]
        sample = str(row.gsm)
        indexed.columns = pd.MultiIndex.from_product(
            [["beta", "coverage", "cpgs"], [sample]],
            names=["metric", "gsm"],
        )
        sample_frames.append(indexed)
    index_columns = ["chromosome", "bin_start"]
    combined = pd.concat(sample_frames, axis=1)
    beta = combined["beta"]
    coverage = combined["coverage"]
    cpgs = combined["cpgs"]
    beta.index.names = index_columns
    coverage.index.names = index_columns
    cpgs.index.names = index_columns
    sample_order = tissue_metadata["gsm"].astype(str).tolist()
    return (
        beta.reindex(columns=sample_order),
        coverage.reindex(columns=sample_order),
        cpgs.reindex(columns=sample_order),
    )


def generate_assignments(
    sample_count: int,
    case_count: int,
) -> tuple[np.ndarray, str]:
    total = math.comb(sample_count, case_count)
    if total <= EMPIRICAL_MAX_ASSIGNMENTS:
        combinations = list(itertools.combinations(range(sample_count), case_count))
        assignments = np.zeros((sample_count, len(combinations)), dtype=float)
        for column, case_indices in enumerate(combinations):
            assignments[list(case_indices), column] = 1.0
        return assignments, f"exact {len(combinations)} assignments"
    rng = np.random.default_rng(SEED + sample_count + case_count)
    assignments = np.zeros(
        (sample_count, EMPIRICAL_MAX_ASSIGNMENTS), dtype=float
    )
    for column in range(EMPIRICAL_MAX_ASSIGNMENTS):
        assignments[
            rng.choice(sample_count, size=case_count, replace=False),
            column,
        ] = 1.0
    return assignments, f"Monte Carlo {EMPIRICAL_MAX_ASSIGNMENTS} assignments"


def empirical_top_window_pvalues(
    values: np.ndarray,
    case_mask: np.ndarray,
) -> tuple[np.ndarray, str]:
    assignments, method = generate_assignments(
        values.shape[1], int(case_mask.sum())
    )
    valid = np.isfinite(values).astype(float)
    filled = np.nan_to_num(values, nan=0.0).astype(float)
    case_sum = filled @ assignments
    case_n = valid @ assignments
    total_sum = filled.sum(axis=1, keepdims=True)
    total_n = valid.sum(axis=1, keepdims=True)
    comparator_sum = total_sum - case_sum
    comparator_n = total_n - case_n
    case_mean = np.divide(
        case_sum,
        case_n,
        out=np.full_like(case_sum, np.nan),
        where=case_n > 0,
    )
    comparator_mean = np.divide(
        comparator_sum,
        comparator_n,
        out=np.full_like(comparator_sum, np.nan),
        where=comparator_n > 0,
    )
    null = np.abs(case_mean - comparator_mean)
    observed = np.abs(
        np.nanmean(values[:, case_mask], axis=1)
        - np.nanmean(values[:, ~case_mask], axis=1)
    )
    exceedances = np.sum(
        null >= observed[:, None] - 1e-12,
        axis=1,
    )
    if method.startswith("exact"):
        exceedances = np.maximum(exceedances, 1)
        pvalue = exceedances / assignments.shape[1]
    else:
        pvalue = (exceedances + 1) / (assignments.shape[1] + 1)
    return pvalue.astype(float), method


def analyze_cohort(
    name: str,
    metadata: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    specification = COHORTS[name]
    tissue_metadata = metadata[
        metadata["tissue"].eq(specification["tissue"])
    ].copy()
    primary = tissue_metadata[
        tissue_metadata["analysis_condition"].isin(
            [specification["case"], specification["comparator"]]
        )
    ].copy()
    beta, coverage, cpgs = build_tissue_matrix(tissue_metadata)
    primary_samples = primary["gsm"].astype(str).tolist()
    beta_primary = beta[primary_samples]
    coverage_primary = coverage[primary_samples]
    cpg_primary = cpgs[primary_samples]
    condition = primary.set_index("gsm")["analysis_condition"].astype(str)
    case_samples = condition[condition == specification["case"]].index.tolist()
    comparator_samples = condition[
        condition == specification["comparator"]
    ].index.tolist()
    minimum_case = max(
        3,
        int(math.ceil(MIN_GROUP_FRACTION * len(case_samples))),
    )
    minimum_comparator = max(
        3,
        int(math.ceil(MIN_GROUP_FRACTION * len(comparator_samples))),
    )
    keep = (
        beta_primary[case_samples].notna().sum(axis=1) >= minimum_case
    ) & (
        beta_primary[comparator_samples].notna().sum(axis=1)
        >= minimum_comparator
    )
    beta_primary = beta_primary.loc[keep]
    coverage_primary = coverage_primary.loc[keep]
    cpg_primary = cpg_primary.loc[keep]
    beta = beta.loc[keep]
    beta_values = beta_primary.to_numpy(dtype=float)
    m_values = np.log2(
        (np.clip(beta_values, 0, 1) + 0.01)
        / (1 - np.clip(beta_values, 0, 1) + 0.01)
    )
    column_lookup = {sample: index for index, sample in enumerate(primary_samples)}
    case_indices = np.asarray([column_lookup[sample] for sample in case_samples])
    comparator_indices = np.asarray(
        [column_lookup[sample] for sample in comparator_samples]
    )
    case_values = m_values[:, case_indices]
    comparator_values = m_values[:, comparator_indices]
    test = stats.ttest_ind(
        case_values,
        comparator_values,
        axis=1,
        equal_var=False,
        nan_policy="omit",
    )
    pvalue = np.asarray(test.pvalue, dtype=float)
    pvalue[~np.isfinite(pvalue)] = 1.0
    beta_case = beta_primary[case_samples].to_numpy(dtype=float)
    beta_comparator = beta_primary[comparator_samples].to_numpy(dtype=float)
    effects = beta_primary.index.to_frame(index=False)
    effects["bin_end"] = effects["bin_start"] + BIN_SIZE
    effects["case_condition"] = specification["case"]
    effects["comparator_condition"] = specification["comparator"]
    effects["case_samples"] = len(case_samples)
    effects["comparator_samples"] = len(comparator_samples)
    effects["case_mean_beta"] = np.nanmean(beta_case, axis=1)
    effects["comparator_mean_beta"] = np.nanmean(beta_comparator, axis=1)
    effects["delta_beta"] = (
        effects["case_mean_beta"] - effects["comparator_mean_beta"]
    )
    effects["hedges_g_m_value"] = hedges_g_matrix(
        case_values, comparator_values
    )
    effects["welch_pvalue"] = pvalue
    effects["welch_fdr"] = bh_adjust(pvalue)
    effects["mean_coverage"] = np.nanmean(
        coverage_primary.to_numpy(dtype=float), axis=1
    )
    effects["median_cpgs"] = np.nanmedian(
        cpg_primary.to_numpy(dtype=float), axis=1
    )
    effects["empirical_pvalue"] = np.nan
    candidate_positions = np.argsort(pvalue)[: min(EMPIRICAL_WINDOWS, len(pvalue))]
    case_mask = np.asarray(
        [sample in set(case_samples) for sample in primary_samples],
        dtype=bool,
    )
    candidate_p, empirical_method = empirical_top_window_pvalues(
        m_values[candidate_positions],
        case_mask,
    )
    effects.loc[candidate_positions, "empirical_pvalue"] = candidate_p
    effects["cohort"] = name
    effects.sort_values(
        ["welch_pvalue", "chromosome", "bin_start"], inplace=True
    )
    output_parquet = ensure_local(
        RESULT_ROOT / f"eclipse_{name}_window_effects.parquet"
    )
    output_csv = ensure_local(
        RESULT_ROOT / f"eclipse_{name}_top_window_effects.csv"
    )
    effects.to_parquet(output_parquet, index=False, compression="zstd")
    effects.head(2_000).to_csv(output_csv, index=False)
    summary = {
        "cohort": name,
        "tissue": specification["tissue"],
        "case_condition": specification["case"],
        "comparator_condition": specification["comparator"],
        "case_samples": len(case_samples),
        "comparator_samples": len(comparator_samples),
        "retained_windows": int(len(effects)),
        "fdr_0_05_windows": int((effects["welch_fdr"] < 0.05).sum()),
        "absolute_delta_0_10_windows": int(
            (effects["delta_beta"].abs() >= 0.10).sum()
        ),
        "minimum_case_samples_per_window": minimum_case,
        "minimum_comparator_samples_per_window": minimum_comparator,
        "empirical_method": empirical_method,
    }
    return effects, beta, summary


def cross_tissue_meta(
    effect_tables: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    cohorts = list(effect_tables)
    metric_frames = []
    cohort_sample_sizes: dict[str, float] = {}
    for cohort, frame in effect_tables.items():
        indexed = frame.set_index(["chromosome", "bin_start"])
        metric = indexed[["delta_beta", "welch_pvalue"]].copy()
        metric.columns = pd.MultiIndex.from_product(
            [[cohort], metric.columns],
            names=["cohort", "metric"],
        )
        metric_frames.append(metric)
        cohort_sample_sizes[cohort] = float(
            frame["case_samples"].iloc[0]
            + frame["comparator_samples"].iloc[0]
        )
    wide = pd.concat(metric_frames, axis=1)
    p = np.column_stack(
        [
            wide[(cohort, "welch_pvalue")].to_numpy(dtype=float)
            for cohort in cohorts
        ]
    )
    delta = np.column_stack(
        [
            wide[(cohort, "delta_beta")].to_numpy(dtype=float)
            for cohort in cohorts
        ]
    )
    finite = np.isfinite(p) & np.isfinite(delta)
    tissue_count = finite.sum(axis=1)
    retain = tissue_count >= 2
    p = p[retain]
    delta = delta[retain]
    finite = finite[retain]
    tissue_count = tissue_count[retain]
    weights = np.asarray(
        [math.sqrt(cohort_sample_sizes[cohort]) for cohort in cohorts],
        dtype=float,
    )
    weighted_mask = finite * weights[None, :]
    clipped_p = np.clip(p, 1e-300, 1.0)
    signed_z = stats.norm.isf(clipped_p / 2) * np.sign(delta)
    signed_z[~finite] = 0.0
    meta_z = np.sum(weighted_mask * signed_z, axis=1) / np.sqrt(
        np.sum(weighted_mask**2, axis=1)
    )
    weighted_delta = np.sum(
        weighted_mask * np.where(finite, delta, 0.0),
        axis=1,
    ) / np.sum(weighted_mask, axis=1)
    delta_mean = np.sum(
        np.where(finite, delta, 0.0), axis=1
    ) / tissue_count
    delta_variance = np.sum(
        np.where(finite, (delta - delta_mean[:, None]) ** 2, 0.0),
        axis=1,
    ) / (tissue_count - 1)
    positive = np.sum(finite & (delta > 0), axis=1)
    negative = np.sum(finite & (delta < 0), axis=1)
    sign_concordance = np.maximum(positive, negative) / tissue_count
    cohort_code = np.sum(
        finite.astype(np.int16)
        * (2 ** np.arange(len(cohorts), dtype=np.int16))[None, :],
        axis=1,
    )
    cohort_lookup = {
        code: "|".join(
            cohort
            for index, cohort in enumerate(cohorts)
            if code & (2**index)
        )
        for code in range(1, 2 ** len(cohorts))
    }
    retained_index = wide.index[retain]
    meta = retained_index.to_frame(index=False)
    meta["bin_end"] = meta["bin_start"].astype(np.int64) + BIN_SIZE
    meta["tissues"] = tissue_count.astype(int)
    meta["meta_z"] = meta_z
    meta["meta_pvalue"] = 2 * stats.norm.sf(np.abs(meta_z))
    meta["weighted_delta_beta"] = weighted_delta
    meta["delta_beta_standard_deviation"] = np.sqrt(delta_variance)
    meta["sign_concordance"] = sign_concordance
    meta["cohorts"] = pd.Series(cohort_code).map(cohort_lookup).to_numpy()
    meta["meta_fdr"] = bh_adjust(meta["meta_pvalue"])
    meta.sort_values(
        ["meta_pvalue", "chromosome", "bin_start"], inplace=True
    )
    meta.to_parquet(
        ensure_local(RESULT_ROOT / "eclipse_cross_tissue_window_meta.parquet"),
        index=False,
        compression="zstd",
    )
    meta.head(5_000).to_csv(
        ensure_local(RESULT_ROOT / "eclipse_cross_tissue_top_windows.csv"),
        index=False,
    )
    return meta


def promoter_effects(
    cohort: str,
    effects: pd.DataFrame,
    genes: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for chromosome, gene_frame in genes.groupby("chromosome", observed=True):
        window = effects[effects["chromosome"] == chromosome].sort_values(
            "bin_start"
        )
        if window.empty:
            continue
        starts = window["bin_start"].to_numpy(dtype=np.int64)
        for gene in gene_frame.itertuples(index=False):
            left = int(
                np.searchsorted(
                    starts,
                    int(gene.promoter_start) - BIN_SIZE + 1,
                    side="left",
                )
            )
            right = int(
                np.searchsorted(starts, int(gene.promoter_end), side="left")
            )
            if right <= left:
                continue
            subset = window.iloc[left:right]
            overlap = subset[
                (subset["bin_start"] < int(gene.promoter_end))
                & (subset["bin_end"] > int(gene.promoter_start))
            ]
            if overlap.empty:
                continue
            rows.append(
                {
                    "cohort": cohort,
                    "gene": gene.gene,
                    "gene_id": gene.gene_id,
                    "gene_type": gene.gene_type,
                    "chromosome": chromosome,
                    "transcription_start": int(gene.transcription_start),
                    "promoter_bins": int(len(overlap)),
                    "promoter_delta_beta": float(
                        np.median(overlap["delta_beta"])
                    ),
                    "promoter_effect_hedges_g": float(
                        np.median(overlap["hedges_g_m_value"])
                    ),
                    "promoter_acat_pvalue": acat(
                        overlap["welch_pvalue"].to_numpy(dtype=float)
                    ),
                    "mean_promoter_coverage": float(
                        overlap["mean_coverage"].mean()
                    ),
                }
            )
    promoter = pd.DataFrame(rows)
    if promoter.empty:
        raise RuntimeError(f"{cohort}: no promoter effects were mapped")
    promoter = (
        promoter.groupby(["cohort", "gene"], as_index=False, observed=True)
        .agg(
            gene_id=(
                "gene_id",
                lambda values: "|".join(
                    sorted(set(values.astype(str)))
                ),
            ),
            gene_type=(
                "gene_type",
                lambda values: "|".join(
                    sorted(set(values.astype(str)))
                ),
            ),
            chromosome=(
                "chromosome",
                lambda values: "|".join(
                    sorted(set(values.astype(str)))
                ),
            ),
            transcription_start=("transcription_start", "median"),
            promoter_bins=("promoter_bins", "sum"),
            promoter_delta_beta=("promoter_delta_beta", "median"),
            promoter_effect_hedges_g=(
                "promoter_effect_hedges_g",
                "median",
            ),
            promoter_acat_pvalue=(
                "promoter_acat_pvalue",
                lambda values: acat(values.to_numpy(dtype=float)),
            ),
            mean_promoter_coverage=("mean_promoter_coverage", "mean"),
        )
    )
    promoter["promoter_fdr"] = bh_adjust(promoter["promoter_acat_pvalue"])
    promoter.sort_values(
        ["promoter_acat_pvalue", "gene"], inplace=True
    )
    promoter.to_parquet(
        ensure_local(RESULT_ROOT / f"eclipse_{cohort}_promoter_effects.parquet"),
        index=False,
        compression="zstd",
    )
    promoter.head(5_000).to_csv(
        ensure_local(RESULT_ROOT / f"eclipse_{cohort}_top_promoters.csv"),
        index=False,
    )
    return promoter


def cross_tissue_promoter_meta(
    promoter_tables: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    concatenated = pd.concat(promoter_tables.values(), ignore_index=True)
    rows: list[dict[str, object]] = []
    for gene, frame in concatenated.groupby("gene", observed=True):
        if len(frame) < 2:
            continue
        p = np.clip(
            frame["promoter_acat_pvalue"].to_numpy(dtype=float),
            1e-300,
            1.0,
        )
        delta = frame["promoter_delta_beta"].to_numpy(dtype=float)
        z = stats.norm.isf(p / 2) * np.sign(delta)
        weights = np.sqrt(frame["promoter_bins"].to_numpy(dtype=float))
        meta_z = float(np.sum(weights * z) / np.sqrt(np.sum(weights**2)))
        rows.append(
            {
                "gene": gene,
                "gene_id": frame["gene_id"].iloc[0],
                "gene_type": frame["gene_type"].iloc[0],
                "tissues": int(len(frame)),
                "meta_z": meta_z,
                "meta_pvalue": float(2 * stats.norm.sf(abs(meta_z))),
                "weighted_delta_beta": float(np.average(delta, weights=weights)),
                "sign_concordance": float(
                    max(np.sum(delta > 0), np.sum(delta < 0)) / len(delta)
                ),
                "cohorts": "|".join(frame["cohort"].astype(str)),
            }
        )
    meta = pd.DataFrame(rows)
    meta["meta_fdr"] = bh_adjust(meta["meta_pvalue"])
    meta.sort_values(["meta_pvalue", "gene"], inplace=True)
    meta.to_csv(
        ensure_local(RESULT_ROOT / "eclipse_cross_tissue_promoter_meta.csv"),
        index=False,
    )
    return meta


def single_cell_integration(
    promoter_meta: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    edge_root = (
        PROJECT_ROOT
        / "."
        / "04_corrected_results"
        / "02_statistical_reanalysis"
    )
    frames: list[pd.DataFrame] = []
    for target_state, state_file in STATE_FILES.items():
        path = edge_root / f"single_cell_pseudobulk_edger_{state_file}.csv"
        expression = pd.read_csv(path)
        expression["target_state"] = target_state
        expression["expression_source_state"] = state_file
        merged = expression.merge(promoter_meta, on="gene", how="inner")
        merged["promoter_activation_effect"] = -merged["weighted_delta_beta"]
        merged["expression_methylation_concordance"] = (
            merged["promoter_activation_effect"] * merged["logFC"]
        )
        merged["direction_concordant"] = (
            np.sign(merged["promoter_activation_effect"])
            == np.sign(merged["logFC"])
        )
        frames.append(merged)
    integrated = pd.concat(frames, ignore_index=True)
    integrated.to_parquet(
        ensure_local(
            RESULT_ROOT / "eclipse_promoter_single_cell_gene_integration.parquet"
        ),
        index=False,
        compression="zstd",
    )

    module_rows: list[dict[str, object]] = []
    for target_state, state_frame in integrated.groupby(
        "target_state", observed=True
    ):
        for module, genes in MODULES.items():
            subset = state_frame[state_frame["gene"].isin(genes)].copy()
            if subset.empty:
                continue
            rho = np.nan
            rho_p = np.nan
            rho_method = "not estimable"
            if len(subset) >= 4:
                rho, rho_p, rho_method = spearman_inference(
                    subset["promoter_activation_effect"].to_numpy(
                        dtype=float
                    ),
                    subset["logFC"].to_numpy(dtype=float),
                )
            module_rows.append(
                {
                    "target_state": target_state,
                    "module": module,
                    "prespecified_genes": len(genes),
                    "mapped_genes": int(subset["gene"].nunique()),
                    "mean_promoter_activation_effect": float(
                        subset["promoter_activation_effect"].mean()
                    ),
                    "mean_expression_logFC": float(subset["logFC"].mean()),
                    "mean_concordance_product": float(
                        subset["expression_methylation_concordance"].mean()
                    ),
                    "direction_concordant_fraction": float(
                        subset["direction_concordant"].mean()
                    ),
                    "spearman_rho": rho,
                    "spearman_pvalue": rho_p,
                    "spearman_pvalue_method": rho_method,
                }
            )
    module_summary = pd.DataFrame(module_rows)
    module_summary["spearman_fdr"] = bh_adjust(module_summary["spearman_pvalue"])
    module_summary.to_csv(
        ensure_local(RESULT_ROOT / "eclipse_cell_state_module_summary.csv"),
        index=False,
    )

    key_genes = sorted({gene for genes in MODULES.values() for gene in genes})
    key_loci = integrated[integrated["gene"].isin(key_genes)].copy()
    key_loci.to_csv(
        ensure_local(RESULT_ROOT / "eclipse_key_locus_single_cell_integration.csv"),
        index=False,
    )
    return integrated, module_summary, key_loci


def stool_projection(
    beta: pd.DataFrame,
    effects: pd.DataFrame,
    metadata: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    stool_metadata = metadata[metadata["tissue"] == "stool"].copy()
    primary = stool_metadata[
        stool_metadata["analysis_condition"].isin(["nec", "control"])
    ].copy()
    selected = effects[
        effects["delta_beta"].abs() >= 0.02
    ].sort_values("welch_pvalue").head(500)
    if len(selected) < 100:
        selected = effects.sort_values("welch_pvalue").head(500)
    selected_index = pd.MultiIndex.from_frame(
        selected[["chromosome", "bin_start"]]
    )
    matrix = beta.reindex(selected_index).T
    primary_samples = primary["gsm"].astype(str).tolist()
    primary_matrix = matrix.loc[primary_samples]
    mean = primary_matrix.mean(axis=0)
    standard_deviation = primary_matrix.std(axis=0, ddof=1).replace(0, np.nan)
    standardized = matrix.sub(mean, axis=1).div(standard_deviation, axis=1)
    standardized = standardized.fillna(0.0)
    condition = primary.set_index("gsm")["analysis_condition"].astype(str)
    axis = (
        standardized.loc[condition[condition == "nec"].index].mean(axis=0)
        - standardized.loc[condition[condition == "control"].index].mean(axis=0)
    ).to_numpy(dtype=float)
    norm = float(np.linalg.norm(axis))
    if not np.isfinite(norm) or norm <= 0:
        raise RuntimeError("Stool methylation NEC axis is not estimable")
    axis /= norm
    scores = standardized.to_numpy(dtype=float) @ axis
    score_frame = stool_metadata.set_index("gsm").loc[matrix.index].reset_index()
    score_frame["eclipse_nec_axis_score"] = scores
    sampling = pd.to_numeric(
        score_frame["days_of_life_at_sampling"], errors="coerce"
    )
    onset = pd.to_numeric(
        score_frame["days_of_life_at_nec_onset"], errors="coerce"
    )
    score_frame["days_relative_to_nec_onset"] = sampling - onset

    loo_scores: dict[str, float] = {}
    primary_beta = beta[primary_samples]
    primary_conditions = primary.set_index("gsm").loc[
        primary_samples, "analysis_condition"
    ].astype(str)
    for held_out in primary_samples:
        training_samples = [
            sample for sample in primary_samples if sample != held_out
        ]
        training_condition = primary_conditions.loc[training_samples]
        train_beta = primary_beta[training_samples]
        train_m = np.log2(
            (np.clip(train_beta.to_numpy(dtype=float), 0, 1) + 0.01)
            / (1 - np.clip(train_beta.to_numpy(dtype=float), 0, 1) + 0.01)
        )
        train_case = training_condition[training_condition == "nec"].index
        train_control = training_condition[training_condition == "control"].index
        column_lookup = {
            sample: index for index, sample in enumerate(training_samples)
        }
        test = stats.ttest_ind(
            train_m[:, [column_lookup[sample] for sample in train_case]],
            train_m[:, [column_lookup[sample] for sample in train_control]],
            axis=1,
            equal_var=False,
            nan_policy="omit",
        )
        fold_p = np.asarray(test.pvalue, dtype=float)
        fold_p[~np.isfinite(fold_p)] = 1.0
        fold_positions = np.argsort(fold_p)[:500]
        train_selected = train_beta.iloc[fold_positions].T
        fold_mean = train_selected.mean(axis=0)
        fold_sd = train_selected.std(axis=0, ddof=1).replace(0, np.nan)
        train_z = train_selected.sub(fold_mean, axis=1).div(fold_sd, axis=1)
        train_z = train_z.fillna(0.0)
        fold_axis = (
            train_z.loc[train_case].mean(axis=0)
            - train_z.loc[train_control].mean(axis=0)
        ).to_numpy(dtype=float)
        fold_norm = float(np.linalg.norm(fold_axis))
        if fold_norm <= 0:
            loo_scores[held_out] = np.nan
            continue
        fold_axis /= fold_norm
        held = primary_beta[held_out].iloc[fold_positions]
        held_z = (
            held.sub(fold_mean).div(fold_sd).replace([np.inf, -np.inf], np.nan)
        ).fillna(0.0)
        loo_scores[held_out] = float(held_z.to_numpy(dtype=float) @ fold_axis)
    score_frame["leave_one_subject_out_score"] = score_frame["gsm"].map(
        loo_scores
    )
    validation = score_frame[
        score_frame["analysis_condition"].isin(["nec", "control"])
    ].dropna(subset=["leave_one_subject_out_score"])
    labels = (validation["analysis_condition"] == "nec").astype(int)
    loo_auc = float(
        roc_auc_score(labels, validation["leave_one_subject_out_score"])
    )

    pca = PCA(n_components=2, random_state=SEED)
    pca_scores = pca.fit_transform(standardized.to_numpy(dtype=float))
    pca_frame = score_frame[
        [
            "gsm",
            "subject_id",
            "analysis_condition",
            "days_relative_to_nec_onset",
        ]
    ].copy()
    pca_frame["PC1"] = pca_scores[:, 0]
    pca_frame["PC2"] = pca_scores[:, 1]
    pca_frame["PC1_variance"] = float(pca.explained_variance_ratio_[0])
    pca_frame["PC2_variance"] = float(pca.explained_variance_ratio_[1])
    score_frame.to_csv(
        ensure_local(RESULT_ROOT / "eclipse_stool_nec_axis_projection.csv"),
        index=False,
    )
    pca_frame.to_csv(
        ensure_local(RESULT_ROOT / "eclipse_stool_methylation_pca.csv"),
        index=False,
    )
    summary = {
        "selected_windows": int(len(selected)),
        "primary_samples": int(len(primary_samples)),
        "future_nec_samples": int(
            (stool_metadata["analysis_condition"] == "future_nec").sum()
        ),
        "future_nec_subjects": int(
            stool_metadata.loc[
                stool_metadata["analysis_condition"] == "future_nec",
                "subject_id",
            ].nunique()
        ),
        "leave_one_subject_out_auc": loo_auc,
        "PC1_variance": float(pca.explained_variance_ratio_[0]),
        "PC2_variance": float(pca.explained_variance_ratio_[1]),
    }
    return score_frame, pca_frame, summary


def main() -> None:
    preparation = json.loads(
        (MANIFESTS / "eclipse_bin_preparation_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    if preparation["status"] != "PASS":
        raise RuntimeError("ECLIPSE bin preparation manifest is not PASS")
    metadata = pd.read_csv(
        PUBLIC_ROOT / "eclipse_sample_metadata.csv",
        low_memory=False,
    )
    effect_tables: dict[str, pd.DataFrame] = {}
    beta_tables: dict[str, pd.DataFrame] = {}
    cohort_summaries: list[dict[str, object]] = []
    for cohort in COHORTS:
        print(f"ECLIPSE_COHORT_START {cohort}", flush=True)
        effects, beta, summary = analyze_cohort(cohort, metadata)
        effect_tables[cohort] = effects
        beta_tables[cohort] = beta
        cohort_summaries.append(summary)
        print(json.dumps(summary), flush=True)

    window_meta = cross_tissue_meta(effect_tables)
    genes = pd.read_parquet(
        PROCESSED_ROOT / "gencode_v46_gene_promoters.parquet"
    )
    promoter_tables = {
        cohort: promoter_effects(cohort, effects, genes)
        for cohort, effects in effect_tables.items()
    }
    promoter_meta = cross_tissue_promoter_meta(promoter_tables)
    integrated, module_summary, key_loci = single_cell_integration(
        promoter_meta
    )
    stool_scores, stool_pca, stool_summary = stool_projection(
        beta_tables["stool"],
        effect_tables["stool"],
        metadata,
    )

    summary_frame = pd.DataFrame(cohort_summaries)
    summary_frame.to_csv(
        ensure_local(RESULT_ROOT / "eclipse_cohort_summary.csv"),
        index=False,
    )
    manifest = {
        "status": "PASS",
        "analysis": "ECLIPSE-NEC",
        "algorithm": (
            "Full-depth 1-kb Bismark count aggregation; tissue-specific "
            "sample-level Welch inference; top-window label permutations; "
            "weighted Stouffer cross-tissue meta-analysis; promoter ACAT; "
            "single-cell expression-methylation concordance; stool "
            "leave-one-subject-out NEC-axis projection"
        ),
        "statistical_unit": "biological sample or subject",
        "cohorts": cohort_summaries,
        "cross_tissue_windows": int(len(window_meta)),
        "cross_tissue_fdr_0_05_windows": int(
            (window_meta["meta_fdr"] < 0.05).sum()
        ),
        "promoter_genes": int(promoter_meta["gene"].nunique()),
        "promoter_fdr_0_05_genes": int(
            (promoter_meta["meta_fdr"] < 0.05).sum()
        ),
        "single_cell_gene_state_pairs": int(len(integrated)),
        "module_state_pairs": int(len(module_summary)),
        "key_locus_state_pairs": int(len(key_loci)),
        "stool_projection": stool_summary,
        "versions": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
        },
        "cloud_files_written": 0,
    }
    (MANIFESTS / "eclipse_analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
