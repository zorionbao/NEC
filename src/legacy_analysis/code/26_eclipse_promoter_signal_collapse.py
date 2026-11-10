#!/usr/bin/env python3
"""Collapse overlapping ECLIPSE promoter annotations into signal units."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ECLIPSE_CODE = ROOT / "code" / "16_eclipse_nec.py"
RESULT_ROOT = ROOT / "results" / "eclipse_nec"
MANIFEST_ROOT = ROOT / "manifests"
PROMOTER_RADIUS = 2_000


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
        "eclipse_signal_source", ECLIPSE_CODE
    )
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def promoter_annotation() -> pd.DataFrame:
    frames = []
    for cohort in ["ileum", "colon", "stool", "blood"]:
        frame = pd.read_parquet(
            RESULT_ROOT / f"eclipse_{cohort}_promoter_effects.parquet",
            columns=["gene", "chromosome", "transcription_start"],
        )
        frames.append(frame)
    annotation = pd.concat(frames, ignore_index=True)
    annotation = (
        annotation.groupby("gene", as_index=False)
        .agg(
            chromosome=(
                "chromosome",
                lambda values: "|".join(
                    sorted(set(values.astype(str)))
                ),
            ),
            transcription_start=("transcription_start", "median"),
        )
    )
    return annotation


def assign_overlapping_signals(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["signal_id"] = ""
    frame["signal_start"] = np.nan
    frame["signal_end"] = np.nan
    unambiguous = frame[
        ~frame["chromosome"].str.contains("|", regex=False)
        & frame["transcription_start"].notna()
    ].copy()
    for chromosome, chromosome_frame in unambiguous.groupby(
        "chromosome", observed=True
    ):
        chromosome_frame = chromosome_frame.sort_values(
            "transcription_start"
        )
        cluster_number = 0
        cluster_end = -np.inf
        cluster_members: list[int] = []
        cluster_start = np.nan

        def close_cluster() -> None:
            if not cluster_members:
                return
            signal = (
                f"{chromosome}:{int(cluster_start)}-{int(cluster_end)}"
            )
            frame.loc[cluster_members, "signal_id"] = signal
            frame.loc[cluster_members, "signal_start"] = cluster_start
            frame.loc[cluster_members, "signal_end"] = cluster_end

        for index, row in chromosome_frame.iterrows():
            start = max(
                0,
                float(row["transcription_start"]) - PROMOTER_RADIUS,
            )
            end = float(row["transcription_start"]) + PROMOTER_RADIUS + 1
            if start > cluster_end:
                close_cluster()
                cluster_number += 1
                cluster_members = [index]
                cluster_start = start
                cluster_end = end
            else:
                cluster_members.append(index)
                cluster_end = max(cluster_end, end)
        close_cluster()

    ambiguous = frame["signal_id"].eq("")
    frame.loc[ambiguous, "signal_id"] = frame.loc[
        ambiguous, "gene"
    ].map(lambda gene: f"ambiguous:{gene}")
    frame.loc[ambiguous, "signal_start"] = frame.loc[
        ambiguous, "transcription_start"
    ]
    frame.loc[ambiguous, "signal_end"] = frame.loc[
        ambiguous, "transcription_start"
    ]
    return frame


def collapse_signals(
    gene_meta: pd.DataFrame,
    eclipse,
) -> pd.DataFrame:
    annotation = assign_overlapping_signals(promoter_annotation())
    merged = gene_meta.merge(
        annotation,
        on="gene",
        how="left",
        validate="one_to_one",
    )
    if merged["signal_id"].isna().any():
        raise RuntimeError("At least one promoter gene lacks a signal ID")
    rows = []
    for signal_id, frame in merged.groupby(
        "signal_id", observed=True, sort=False
    ):
        ranked = frame.assign(
            type_priority=~frame["gene_type"].eq("protein_coding")
        ).sort_values(
            ["meta_pvalue", "type_priority", "gene"]
        )
        representative = ranked.iloc[0]
        rows.append(
            {
                "signal_id": signal_id,
                "chromosome": representative["chromosome"],
                "signal_start": float(frame["signal_start"].min()),
                "signal_end": float(frame["signal_end"].max()),
                "gene_count": int(frame["gene"].nunique()),
                "genes": "|".join(sorted(frame["gene"].astype(str))),
                "representative_gene": str(representative["gene"]),
                "representative_gene_type": str(
                    representative["gene_type"]
                ),
                "tissues": int(frame["tissues"].max()),
                "signal_acat_pvalue": eclipse.acat(
                    frame["meta_pvalue"].to_numpy(dtype=float)
                ),
                "median_weighted_delta_beta": float(
                    frame["weighted_delta_beta"].median()
                ),
                "minimum_sign_concordance": float(
                    frame["sign_concordance"].min()
                ),
            }
        )
    signals = pd.DataFrame(rows)
    signals["signal_fdr"] = eclipse.bh_adjust(
        signals["signal_acat_pvalue"]
    )
    signals.sort_values(
        ["signal_acat_pvalue", "signal_id"], inplace=True
    )
    return signals


def update_nested_counts(payload: object, signal_count: int) -> None:
    if isinstance(payload, dict):
        if "promoter_fdr_0_05_genes" in payload:
            payload["promoter_gene_level_fdr_0_05"] = payload.pop(
                "promoter_fdr_0_05_genes"
            )
            payload["promoter_signal_fdr_0_05"] = signal_count
        for value in payload.values():
            update_nested_counts(value, signal_count)
    elif isinstance(payload, list):
        for value in payload:
            update_nested_counts(value, signal_count)


def main() -> None:
    eclipse = load_eclipse_module()
    gene_meta_path = (
        RESULT_ROOT / "eclipse_cross_tissue_promoter_meta.csv"
    )
    gene_meta = pd.read_csv(gene_meta_path)
    signals = collapse_signals(gene_meta, eclipse)
    output_path = ensure_local(
        RESULT_ROOT / "eclipse_cross_tissue_promoter_signal_meta.csv"
    )
    signals.to_csv(output_path, index=False)

    _, module_summary, _ = eclipse.single_cell_integration(gene_meta)
    signal_significant = int(signals["signal_fdr"].lt(0.05).sum())
    gene_significant = int(gene_meta["meta_fdr"].lt(0.05).sum())
    module_significant = int(
        module_summary["spearman_fdr"].lt(0.05).sum()
    )

    main_manifest_path = MANIFEST_ROOT / "eclipse_analysis_manifest.json"
    main_manifest = json.loads(
        main_manifest_path.read_text(encoding="utf-8")
    )
    update_nested_counts(main_manifest, signal_significant)
    main_manifest["promoter_annotation_genes"] = int(
        gene_meta["gene"].nunique()
    )
    main_manifest["promoter_unique_signals"] = int(len(signals))
    main_manifest["promoter_gene_level_fdr_0_05"] = gene_significant
    main_manifest["promoter_signal_fdr_0_05"] = signal_significant
    main_manifest["module_spearman_fdr_0_05"] = module_significant
    main_manifest_path.write_text(
        json.dumps(main_manifest, indent=2),
        encoding="utf-8",
    )

    manifest = {
        "status": "PASS",
        "analysis": "ECLIPSE overlapping promoter signal collapse",
        "promoter_radius_bp": PROMOTER_RADIUS,
        "annotated_genes": int(gene_meta["gene"].nunique()),
        "unique_overlapping_promoter_signals": int(len(signals)),
        "gene_level_fdr_0_05": gene_significant,
        "signal_level_fdr_0_05": signal_significant,
        "module_correlations_fdr_0_05_after_exact_small_n_tests": (
            module_significant
        ),
        "minimum_signal_pvalue": float(
            signals["signal_acat_pvalue"].min()
        ),
        "minimum_signal_fdr": float(signals["signal_fdr"].min()),
        "cloud_files_written": 0,
    }
    ensure_local(
        MANIFEST_ROOT / "eclipse_promoter_signal_manifest.json"
    ).write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print(
        "ECLIPSE_PROMOTER_SIGNAL_COMPLETE "
        f"signals={len(signals)} "
        f"gene_fdr={gene_significant} "
        f"signal_fdr={signal_significant} "
        f"module_fdr={module_significant}",
        flush=True,
    )


if __name__ == "__main__":
    main()
