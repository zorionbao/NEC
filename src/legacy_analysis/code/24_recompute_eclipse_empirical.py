#!/usr/bin/env python3
"""Recompute ECLIPSE top-window permutations in float64.

This targeted repair reads only the 500 prespecified top Welch windows from
each completed tissue analysis and extracts those windows from each sample
parquet. It replaces the empirical P-value column without repeating the
genome-wide Welch tests.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ECLIPSE_CODE = ROOT / "code" / "16_eclipse_nec.py"
PUBLIC_ROOT = ROOT / "data" / "public" / "eclipse_nec"
BIN_ROOT = ROOT / "data" / "processed" / "eclipse_nec" / "bin_1kb"
RESULT_ROOT = ROOT / "results" / "eclipse_nec"
MANIFEST_ROOT = ROOT / "manifests"


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
        "eclipse_repair_source", ECLIPSE_CODE
    )
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def candidate_matrix(
    candidates: pd.DataFrame,
    samples: list[str],
    eclipse,
) -> np.ndarray:
    candidate_index = pd.MultiIndex.from_frame(
        candidates[["chromosome", "bin_start"]]
    )
    columns = []
    for sample in samples:
        frame = pd.read_parquet(
            BIN_ROOT / f"{sample}_bin_1kb.parquet",
            columns=["chromosome", "bin_start", "coverage", "cpgs", "beta"],
        )
        frame = frame[
            (frame["coverage"] >= eclipse.MIN_COVERAGE)
            & (frame["cpgs"] >= eclipse.MIN_CPGS)
        ]
        series = frame.set_index(["chromosome", "bin_start"])["beta"]
        columns.append(series.reindex(candidate_index).to_numpy(dtype=float))
    return np.column_stack(columns)


def main() -> None:
    eclipse = load_eclipse_module()
    metadata = pd.read_csv(
        PUBLIC_ROOT / "eclipse_sample_metadata.csv",
        low_memory=False,
    )
    summary_rows = []
    for cohort, cohort_specification in eclipse.COHORTS.items():
        if cohort == "stool":
            summary_rows.append(
                {
                    "cohort": cohort,
                    "samples": 13,
                    "case_samples": 8,
                    "recomputed_windows": 0,
                    "permutation_method": (
                        "not applied to covariate-adjusted OLS"
                    ),
                    "minimum_empirical_pvalue": np.nan,
                }
            )
            continue
        effect_path = RESULT_ROOT / f"eclipse_{cohort}_window_effects.parquet"
        if not effect_path.exists():
            raise FileNotFoundError(effect_path)
        effects = pd.read_parquet(effect_path)
        candidates = effects.nsmallest(
            min(eclipse.EMPIRICAL_WINDOWS, len(effects)),
            "welch_pvalue",
        ).copy()

        tissue = metadata[
            metadata["tissue"].eq(cohort_specification["tissue"])
        ].copy()
        primary = tissue[
            tissue["analysis_condition"].isin(
                [
                    cohort_specification["case"],
                    cohort_specification["comparator"],
                ]
            )
        ].copy()
        samples = primary["gsm"].astype(str).tolist()
        condition = primary.set_index("gsm")[
            "analysis_condition"
        ].astype(str)
        case_samples = condition[
            condition.eq(cohort_specification["case"])
        ].index.astype(str).tolist()
        case_mask = np.asarray(
            [sample in set(case_samples) for sample in samples],
            dtype=bool,
        )

        beta = candidate_matrix(candidates, samples, eclipse)
        m_values = np.log2(
            (np.clip(beta, 0, 1) + 0.01)
            / (1 - np.clip(beta, 0, 1) + 0.01)
        )
        empirical, method = eclipse.empirical_top_window_pvalues(
            m_values, case_mask
        )
        replacement = candidates[
            ["chromosome", "bin_start"]
        ].copy()
        replacement["empirical_pvalue_recomputed"] = empirical
        repaired = effects.merge(
            replacement,
            on=["chromosome", "bin_start"],
            how="left",
            validate="one_to_one",
        )
        repaired["empirical_pvalue"] = repaired[
            "empirical_pvalue_recomputed"
        ].combine_first(repaired["empirical_pvalue"])
        repaired.drop(
            columns=["empirical_pvalue_recomputed"],
            inplace=True,
        )
        repaired.sort_values(
            ["welch_pvalue", "chromosome", "bin_start"],
            inplace=True,
        )
        repaired.to_parquet(
            ensure_local(effect_path),
            index=False,
            compression="zstd",
        )
        repaired.head(2_000).to_csv(
            ensure_local(
                RESULT_ROOT
                / f"eclipse_{cohort}_top_window_effects.csv"
            ),
            index=False,
        )
        minimum = float(np.nanmin(empirical))
        if minimum <= 0:
            raise RuntimeError(
                f"{cohort}: empirical P-value repair retained zero"
            )
        summary_rows.append(
            {
                "cohort": cohort,
                "samples": len(samples),
                "case_samples": int(case_mask.sum()),
                "recomputed_windows": len(candidates),
                "permutation_method": method,
                "minimum_empirical_pvalue": minimum,
            }
        )
        print(
            f"ECLIPSE_EMPIRICAL_REPAIRED {cohort} "
            f"windows={len(candidates)} min_p={minimum:.8g}",
            flush=True,
        )

    summary = pd.DataFrame(summary_rows)
    summary_path = ensure_local(
        RESULT_ROOT / "eclipse_empirical_recomputation.csv"
    )
    summary.to_csv(summary_path, index=False)
    manifest = {
        "status": "PASS",
        "analysis": "ECLIPSE top-window empirical P-value recomputation",
        "reason": (
            "float64 recomputation prevents float32 underflow below the "
            "minimum attainable exact permutation P-value"
        ),
        "cohorts": summary_rows,
        "zero_empirical_pvalues_after_repair": 0,
        "cloud_files_written": 0,
    }
    ensure_local(
        MANIFEST_ROOT / "eclipse_empirical_recomputation_manifest.json"
    ).write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print("ECLIPSE_EMPIRICAL_RECOMPUTATION_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
