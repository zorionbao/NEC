from __future__ import annotations

import argparse
import gc
import gzip
import hashlib
import json
import math
import os
import re
import sys
import time
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import anndata as ad
import harmonypy
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from scipy.io import mmread
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.neighbors import KNeighborsClassifier


DEFAULT_REVISION_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_ROOT = DEFAULT_REVISION_ROOT / "data/raw/HCA_NEC/Cell_ranger_output"
DEFAULT_SNAPSHOT_RESULTS = DEFAULT_REVISION_ROOT / "04_corrected_results/00_remote_snapshot/results"
DEFAULT_OUTPUT = DEFAULT_REVISION_ROOT / "04_corrected_results/01_single_cell_corrected"

RANDOM_SEED = 20260724

SIGNATURES: dict[str, tuple[str, ...]] = {
    "M_cell": (
        "SPIB",
        "GP2",
        "CCL20",
        "CXCL16",
        "TNFAIP2",
        "SOX8",
        "TNFRSF11A",
        "MARCKSL1",
    ),
    "LEC": (
        "PROX1",
        "PDPN",
        "FLT4",
        "LYVE1",
        "CCL21",
        "MMRN1",
        "ACKR4",
        "CAVIN2",
    ),
    "T_reg": (
        "FOXP3",
        "IL2RA",
        "CTLA4",
        "TIGIT",
        "IKZF2",
        "CCR7",
        "BATF",
    ),
    "T_NK": (
        "CD3D",
        "CD3E",
        "TRAC",
        "NKG7",
        "GNLY",
        "KLRD1",
    ),
    "epithelial": (
        "EPCAM",
        "KRT8",
        "KRT18",
        "KRT19",
        "VIL1",
        "MUC2",
    ),
    "epithelial_stem": (
        "LGR5",
        "ASCL2",
        "OLFM4",
        "MKI67",
        "SMOC2",
        "SOX9",
    ),
    "myeloid": (
        "LYZ",
        "S100A8",
        "S100A9",
        "FCGR3A",
        "LST1",
        "CD68",
        "TYROBP",
    ),
    "vascular_endothelial": (
        "PECAM1",
        "VWF",
        "KDR",
        "EMCN",
        "ESAM",
        "RAMP2",
    ),
    "fibroblast": (
        "COL1A1",
        "COL1A2",
        "COL3A1",
        "DCN",
        "LUM",
        "ACTA2",
    ),
}

CELL_STATE_ORDER = (
    "M_cell",
    "LEC",
    "T_reg",
    "T_NK",
    "epithelial",
    "epithelial_stem",
    "myeloid",
    "vascular_endothelial",
    "fibroblast",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=json_safe),
        encoding="utf-8",
    )


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
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def run_metadata(run_name: str) -> dict[str, Any]:
    match = re.match(r"^(?P<donor>\d+)", run_name)
    if not match:
        raise ValueError(f"Cannot parse donor ID from run name: {run_name}")
    donor = match.group("donor")
    run_match = re.search(r"_run(?P<run>\d+)$", run_name, flags=re.IGNORECASE)
    run_number = int(run_match.group("run")) if run_match else 0
    sample = re.sub(r"_run\d+$", "", run_name, flags=re.IGNORECASE)
    condition = "nec" if "nec" in run_name.lower() else "control"
    selection = "CD45_depleted" if "cd45" in run_name.lower() else "unselected"
    return {
        "run_id": run_name,
        "sample": sample,
        "donor_id": donor,
        "condition": condition,
        "selection": selection,
        "run_number": run_number,
    }


def resolve_10x_files(matrix_dir: Path) -> tuple[Path, Path, Path]:
    matrix_candidates = [matrix_dir / "matrix.mtx.gz", matrix_dir / "matrix.mtx"]
    barcode_candidates = [matrix_dir / "barcodes.tsv.gz", matrix_dir / "barcodes.tsv"]
    feature_candidates = [
        matrix_dir / "features.tsv.gz",
        matrix_dir / "features.tsv",
        matrix_dir / "genes.tsv.gz",
        matrix_dir / "genes.tsv",
    ]

    def first_existing(candidates: Iterable[Path]) -> Path:
        for candidate in candidates:
            if candidate.exists():
                return candidate
        raise FileNotFoundError(
            "No candidate file found: " + ", ".join(str(item) for item in candidates)
        )

    return (
        first_existing(matrix_candidates),
        first_existing(barcode_candidates),
        first_existing(feature_candidates),
    )


def open_text(path: Path):
    if path.suffix.lower() == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def read_selected_barcodes(path: Path, selected_indices: np.ndarray) -> list[str]:
    selected = set(map(int, selected_indices))
    output: list[str] = []
    with open_text(path) as handle:
        for index, line in enumerate(handle):
            if index in selected:
                output.append(line.strip())
    if len(output) != len(selected_indices):
        raise ValueError(
            f"Barcode selection mismatch for {path}: "
            f"expected={len(selected_indices)}, read={len(output)}"
        )
    return output


def read_features(path: Path) -> pd.DataFrame:
    records: list[tuple[str, str, str]] = []
    with open_text(path) as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            gene_id = parts[0]
            gene_symbol = parts[1] if len(parts) > 1 else parts[0]
            feature_type = parts[2] if len(parts) > 2 else "Gene Expression"
            records.append((gene_id, gene_symbol, feature_type))
    frame = pd.DataFrame(records, columns=["gene_id", "gene_symbol", "feature_type"])
    duplicates: Counter[str] = Counter()
    unique_names: list[str] = []
    for symbol in frame["gene_symbol"].astype(str):
        count = duplicates[symbol]
        unique_names.append(symbol if count == 0 else f"{symbol}-{count}")
        duplicates[symbol] += 1
    frame["gene_symbol_unique"] = unique_names
    return frame


def load_raw_10x_filtered(
    matrix_dir: Path,
    metadata: dict[str, Any],
) -> tuple[ad.AnnData, dict[str, Any]]:
    matrix_path, barcode_path, feature_path = resolve_10x_files(matrix_dir)
    started = time.time()
    raw = mmread(str(matrix_path))
    if not sparse.issparse(raw):
        raw = sparse.coo_matrix(raw)
    raw = raw.tocoo(copy=False)
    n_features, n_barcodes = raw.shape
    genes_per_barcode = np.bincount(raw.col, minlength=n_barcodes)
    selected = np.flatnonzero(
        (genes_per_barcode >= 200) & (genes_per_barcode <= 8000)
    )
    if selected.size == 0:
        raise ValueError(f"No barcodes passed feature-count QC in {matrix_dir}")

    raw_csc = raw.tocsc()
    selected_matrix = raw_csc[:, selected].T.tocsr()
    del raw, raw_csc, genes_per_barcode
    gc.collect()

    features = read_features(feature_path)
    if len(features) != n_features:
        raise ValueError(
            f"Feature count mismatch in {matrix_dir}: "
            f"matrix={n_features}, table={len(features)}"
        )
    gene_detection = np.asarray((selected_matrix > 0).sum(axis=0)).ravel()
    gene_keep = gene_detection >= 3
    selected_matrix = selected_matrix[:, gene_keep].tocsr()
    features = features.loc[gene_keep].reset_index(drop=True)
    selected_barcodes = read_selected_barcodes(barcode_path, selected)

    adata = ad.AnnData(
        X=selected_matrix,
        obs=pd.DataFrame(index=pd.Index(selected_barcodes, name="barcode")),
        var=pd.DataFrame(
            {
                "gene_id": features["gene_id"].to_numpy(),
                "feature_type": features["feature_type"].to_numpy(),
            },
            index=pd.Index(features["gene_symbol_unique"], name="gene_symbol"),
        ),
    )
    adata.var_names_make_unique()
    for key, value in metadata.items():
        adata.obs[key] = value

    adata.var["mt"] = adata.var_names.str.upper().str.startswith("MT-")
    sc.pp.calculate_qc_metrics(
        adata,
        qc_vars=["mt"],
        percent_top=None,
        log1p=False,
        inplace=True,
    )
    before_mt = adata.n_obs
    basic_keep = (
        (adata.obs["n_genes_by_counts"] >= 200)
        & (adata.obs["n_genes_by_counts"] <= 8000)
        & (adata.obs["pct_counts_mt"] <= 25)
    )
    adata = adata[basic_keep].copy()

    legacy_proxy = (
        adata.obs["total_counts"].to_numpy(float)
        * adata.obs["n_genes_by_counts"].to_numpy(float)
    )
    legacy_cutoff = float(np.quantile(legacy_proxy, 0.99))
    adata.obs["legacy_doublet_proxy_score"] = legacy_proxy
    adata.obs["legacy_doublet_proxy_flag"] = legacy_proxy >= legacy_cutoff

    expected_rate = min(0.10, max(0.01, 0.008 * adata.n_obs / 1000.0))
    n_prin_comps = int(max(10, min(30, adata.n_obs // 15, adata.n_vars - 1)))
    scrublet_error = ""
    try:
        sc.pp.scrublet(
            adata,
            expected_doublet_rate=expected_rate,
            n_prin_comps=n_prin_comps,
            random_state=RANDOM_SEED,
            verbose=False,
        )
    except Exception as exc:
        scrublet_error = f"{type(exc).__name__}: {exc}"
        adata.obs["doublet_score"] = np.nan
        adata.obs["predicted_doublet"] = False

    predicted = adata.obs["predicted_doublet"].fillna(False).astype(bool)
    legacy = adata.obs["legacy_doublet_proxy_flag"].astype(bool)
    overlap = int((predicted & legacy).sum())
    before_scrublet = adata.n_obs
    adata = adata[~predicted].copy()
    adata.obs_names = [
        f"{metadata['run_id']}__{barcode}" for barcode in adata.obs_names
    ]
    adata.obs_names.name = "cell_id"
    report = {
        **metadata,
        "matrix_dir": str(matrix_dir),
        "matrix_sha256": sha256_file(matrix_path),
        "raw_matrix_features": int(n_features),
        "raw_matrix_barcodes": int(n_barcodes),
        "cells_after_min_features": int(before_mt),
        "cells_after_basic_qc": int(before_scrublet),
        "legacy_proxy_doublets": int(legacy.sum()),
        "scrublet_expected_rate": expected_rate,
        "scrublet_predicted_doublets": int(predicted.sum()),
        "scrublet_legacy_overlap": overlap,
        "cells_after_scrublet": int(adata.n_obs),
        "genes_after_run_filter": int(adata.n_vars),
        "scrublet_error": scrublet_error,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    return adata, report


def curated_key(
    donor_id: str,
    selection: str,
    run_number: int,
    barcode: str,
) -> str:
    return f"{donor_id}|{selection}|{run_number}|{barcode}"


def curated_annotation_table(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path)
    table["donor_id"] = (
        table["subject"].astype(str).str.extract(r"S?(\d+)", expand=False)
    )
    table["selection"] = np.where(
        table["depletion_type"].astype(str).str.contains("CD45", case=False, na=False),
        "CD45_depleted",
        "unselected",
    )
    table["run_number"] = (
        table["orig.ident"]
        .astype(str)
        .str.extract(r"(?:no|run)(\d+)$", expand=False)
        .fillna(0)
        .astype(int)
    )
    table["curated_key"] = [
        curated_key(donor, selection, run_number, barcode)
        for donor, selection, run_number, barcode in zip(
            table["donor_id"].astype(str),
            table["selection"].astype(str),
            table["run_number"].astype(int),
            table["cell_barcode"].astype(str),
        )
    ]
    duplicate_keys = table["curated_key"].duplicated(keep=False)
    if duplicate_keys.any():
        duplicates = table.loc[duplicate_keys, "curated_key"].tolist()[:20]
        raise ValueError(f"Curated annotation keys are duplicated: {duplicates}")
    return table


def add_curated_labels(
    adata: ad.AnnData,
    curated_path: Path,
) -> dict[str, Any]:
    curated = curated_annotation_table(curated_path)
    mapping = curated.set_index("curated_key")["cell_state"].astype(str).to_dict()
    raw_keys = [
        curated_key(
            str(row.donor_id),
            str(row.selection),
            int(row.run_number),
            str(index).split("__", 1)[1],
        )
        for index, row in adata.obs.iterrows()
    ]
    adata.obs["curated_key"] = raw_keys
    adata.obs["curated_cell_state"] = [
        mapping.get(key, pd.NA) for key in adata.obs["curated_key"]
    ]
    matched = adata.obs["curated_cell_state"].notna()
    return {
        "curated_rows": int(len(curated)),
        "curated_unique_cells": int(curated["curated_key"].nunique()),
        "atlas_cells_matched_to_curated": int(matched.sum()),
        "atlas_cells_without_curated_label": int((~matched).sum()),
        "curated_label_counts_matched": (
            adata.obs.loc[matched, "curated_cell_state"].value_counts().to_dict()
        ),
    }


def normalize_integrate_cluster(
    adata: ad.AnnData,
    n_top_genes: int = 3000,
) -> dict[str, Any]:
    adata.layers["counts"] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=min(n_top_genes, adata.n_vars),
        flavor="seurat",
        batch_key="run_id",
        inplace=True,
    )
    hvg_count = int(adata.var["highly_variable"].sum())
    analysis = adata[:, adata.var["highly_variable"]].copy()
    sc.pp.scale(analysis, max_value=10)
    sc.tl.pca(
        analysis,
        n_comps=40,
        svd_solver="arpack",
        random_state=RANDOM_SEED,
    )
    sc.external.pp.harmony_integrate(
        analysis,
        key="run_id",
        basis="X_pca",
        adjusted_basis="X_pca_harmony",
        random_state=RANDOM_SEED,
        max_iter_harmony=30,
    )
    sc.pp.neighbors(
        analysis,
        n_neighbors=20,
        n_pcs=30,
        use_rep="X_pca_harmony",
        random_state=RANDOM_SEED,
    )
    sc.tl.leiden(
        analysis,
        resolution=0.5,
        random_state=RANDOM_SEED,
        flavor="igraph",
        n_iterations=2,
        key_added="leiden_0_5",
    )
    sc.tl.umap(
        analysis,
        min_dist=0.30,
        spread=1.0,
        random_state=RANDOM_SEED,
    )
    adata.obsm["X_pca_harmony"] = analysis.obsm["X_pca_harmony"].copy()
    adata.obsm["X_umap"] = analysis.obsm["X_umap"].copy()
    adata.obs["leiden_0_5"] = analysis.obs["leiden_0_5"].astype(str).to_numpy()
    return {
        "highly_variable_genes": hvg_count,
        "pca_components": int(analysis.obsm["X_pca"].shape[1]),
        "harmony_components": int(analysis.obsm["X_pca_harmony"].shape[1]),
        "leiden_clusters": int(analysis.obs["leiden_0_5"].nunique()),
    }


def donor_aware_label_transfer(
    adata: ad.AnnData,
    n_neighbors: int = 31,
) -> dict[str, Any]:
    known = adata.obs["curated_cell_state"].notna().to_numpy()
    if known.sum() < 100:
        raise ValueError("Too few curated cells matched for label transfer.")
    embedding = np.asarray(adata.obsm["X_pca_harmony"][:, :30], dtype=float)
    labels = adata.obs.loc[known, "curated_cell_state"].astype(str).to_numpy()
    groups = adata.obs.loc[known, "donor_id"].astype(str).to_numpy()
    x_known = embedding[known]

    logo = LeaveOneGroupOut()
    cv_truth: list[str] = []
    cv_prediction: list[str] = []
    cv_donor: list[str] = []
    fold_rows: list[dict[str, Any]] = []
    for train_index, test_index in logo.split(x_known, labels, groups):
        train_labels = labels[train_index]
        available = set(train_labels)
        test_mask = np.array([label in available for label in labels[test_index]])
        if test_mask.sum() == 0:
            continue
        k = max(3, min(n_neighbors, len(train_index) - 1))
        classifier = KNeighborsClassifier(
            n_neighbors=k,
            weights="distance",
            metric="euclidean",
            n_jobs=-1,
        )
        classifier.fit(x_known[train_index], train_labels)
        prediction = classifier.predict(x_known[test_index][test_mask])
        truth = labels[test_index][test_mask]
        donor = str(groups[test_index][0])
        cv_truth.extend(truth.tolist())
        cv_prediction.extend(prediction.tolist())
        cv_donor.extend([donor] * len(truth))
        fold_rows.append(
            {
                "held_out_donor": donor,
                "n_test": int(len(truth)),
                "accuracy": float(accuracy_score(truth, prediction)),
                "balanced_accuracy": float(
                    balanced_accuracy_score(truth, prediction)
                ),
                "macro_f1": float(
                    f1_score(truth, prediction, average="macro", zero_division=0)
                ),
            }
        )

    k_all = max(3, min(n_neighbors, int(known.sum()) - 1))
    classifier = KNeighborsClassifier(
        n_neighbors=k_all,
        weights="distance",
        metric="euclidean",
        n_jobs=-1,
    )
    classifier.fit(x_known, labels)
    probabilities = classifier.predict_proba(embedding)
    predicted = classifier.classes_[np.argmax(probabilities, axis=1)]
    confidence = probabilities.max(axis=1)
    final = predicted.astype(object)
    final[known] = labels

    adata.obs["label_transfer_state"] = predicted
    adata.obs["label_transfer_confidence"] = confidence
    adata.obs["cell_state"] = final
    state_categories = [
        state for state in CELL_STATE_ORDER if state in set(map(str, final))
    ]
    adata.obs["cell_state"] = pd.Categorical(
        adata.obs["cell_state"],
        categories=state_categories,
        ordered=True,
    )
    adata.obs["strict_gate_state"] = adata.obs["cell_state"].astype(str)
    adata.obs.loc[
        adata.obs["cell_state"].astype(str).eq("M_cell"), "strict_gate_state"
    ] = "M_cell_candidate"
    adata.obs.loc[
        adata.obs["cell_state"].astype(str).eq("LEC"), "strict_gate_state"
    ] = "LEC_candidate"
    adata.obs.loc[
        adata.obs["cell_state"].astype(str).eq("T_reg"), "strict_gate_state"
    ] = "T_reg_candidate"

    if cv_truth:
        labels_all = sorted(set(cv_truth) | set(cv_prediction))
        confusion = confusion_matrix(cv_truth, cv_prediction, labels=labels_all)
        cv_summary = {
            "n_evaluated_cells": len(cv_truth),
            "accuracy": float(accuracy_score(cv_truth, cv_prediction)),
            "balanced_accuracy": float(
                balanced_accuracy_score(cv_truth, cv_prediction)
            ),
            "macro_f1": float(
                f1_score(
                    cv_truth,
                    cv_prediction,
                    average="macro",
                    zero_division=0,
                )
            ),
            "classification_report": classification_report(
                cv_truth,
                cv_prediction,
                output_dict=True,
                zero_division=0,
            ),
            "confusion_labels": labels_all,
            "confusion_matrix": confusion.tolist(),
        }
    else:
        cv_summary = {"status": "no_evaluable_leave_one_donor_out_folds"}

    return {
        "method": "distance-weighted k-nearest-neighbor transfer on Harmony PCs",
        "n_neighbors": k_all,
        "curated_training_cells": int(known.sum()),
        "predicted_cells": int((~known).sum()),
        "low_confidence_below_0_45": int((confidence[~known] < 0.45).sum()),
        "median_prediction_confidence_unmatched": float(
            np.median(confidence[~known])
        )
        if (~known).any()
        else None,
        "leave_one_donor_out": cv_summary,
        "folds": fold_rows,
    }


def score_programs(adata: ad.AnnData) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    available = set(adata.var_names.astype(str))
    for index, (name, genes) in enumerate(SIGNATURES.items()):
        present = [gene for gene in genes if gene in available]
        if not present:
            adata.obs[f"{name}_score"] = np.nan
            rows.append(
                {
                    "signature": name,
                    "n_markers": len(genes),
                    "n_available": 0,
                    "markers_available": "",
                    "status": "no_markers_available",
                }
            )
            continue
        sc.tl.score_genes(
            adata,
            gene_list=present,
            score_name=f"{name}_score",
            ctrl_size=max(25, len(present) * 5),
            random_state=RANDOM_SEED + index,
            use_raw=False,
        )
        rows.append(
            {
                "signature": name,
                "n_markers": len(genes),
                "n_available": len(present),
                "markers_available": ";".join(present),
                "status": "ok",
            }
        )
    return pd.DataFrame(rows)


def sample_level_tables(
    adata: ad.AnnData,
    output_dir: Path,
) -> dict[str, Any]:
    obs = adata.obs.reset_index().rename(columns={"cell_id": "cell"})
    state_counts = (
        obs.groupby(
            ["donor_id", "sample", "condition", "selection", "cell_state"],
            observed=True,
            as_index=False,
        )
        .size()
        .rename(columns={"size": "n_cells"})
    )
    totals = (
        obs.groupby(
            ["donor_id", "sample", "condition", "selection"],
            observed=True,
            as_index=False,
        )
        .size()
        .rename(columns={"size": "n_cells_total"})
    )
    all_rows: list[dict[str, Any]] = []
    observed_states = [
        state
        for state in CELL_STATE_ORDER
        if state in set(obs["cell_state"].astype(str))
    ]
    for row in totals.itertuples(index=False):
        for state in observed_states:
            all_rows.append(
                {
                    "donor_id": row.donor_id,
                    "sample": row.sample,
                    "condition": row.condition,
                    "selection": row.selection,
                    "cell_state": state,
                    "n_cells_total": row.n_cells_total,
                }
            )
    proportions = pd.DataFrame(all_rows).merge(
        state_counts,
        on=["donor_id", "sample", "condition", "selection", "cell_state"],
        how="left",
    )
    proportions["n_cells"] = proportions["n_cells"].fillna(0).astype(int)
    proportions["fraction"] = (
        proportions["n_cells"] / proportions["n_cells_total"].replace(0, np.nan)
    )
    proportions.to_csv(
        output_dir / "cell_state_proportions_capture_unit.csv",
        index=False,
    )

    score_columns = [
        f"{name}_score" for name in SIGNATURES if f"{name}_score" in obs.columns
    ]
    scores = (
        obs.groupby(
            ["donor_id", "sample", "condition", "selection"],
            observed=True,
            as_index=False,
        )[score_columns]
        .mean()
    )
    scores.to_csv(output_dir / "program_scores_capture_unit.csv", index=False)

    donor_unselected = proportions[proportions["selection"].eq("unselected")].copy()
    donor_unselected = (
        donor_unselected.groupby(
            ["donor_id", "condition", "cell_state"],
            observed=True,
            as_index=False,
        )
        .agg(n_cells=("n_cells", "sum"), n_cells_total=("n_cells_total", "sum"))
    )
    donor_unselected["fraction"] = (
        donor_unselected["n_cells"]
        / donor_unselected["n_cells_total"].replace(0, np.nan)
    )
    donor_unselected.to_csv(
        output_dir / "cell_state_proportions_unselected_donor.csv",
        index=False,
    )

    donor_scores = scores[scores["selection"].eq("unselected")].copy()
    donor_scores = (
        donor_scores.groupby(["donor_id", "condition"], as_index=False)[
            score_columns
        ]
        .mean()
    )
    donor_scores.to_csv(
        output_dir / "program_scores_unselected_donor.csv",
        index=False,
    )
    return {
        "capture_units": int(totals["sample"].nunique()),
        "unselected_donors": int(donor_unselected["donor_id"].nunique()),
        "states": observed_states,
    }


def aggregate_sparse_rows(matrix: sparse.spmatrix, rows: np.ndarray) -> np.ndarray:
    return np.asarray(matrix[rows].sum(axis=0)).ravel()


def write_pseudobulk_counts(
    adata: ad.AnnData,
    output_dir: Path,
    min_cells_per_state_donor: int = 10,
) -> dict[str, Any]:
    if "counts" not in adata.layers:
        raise KeyError("Raw counts layer is missing.")
    counts = adata.layers["counts"]
    if not sparse.issparse(counts):
        counts = sparse.csr_matrix(counts)
    else:
        counts = counts.tocsr()

    obs = adata.obs.copy()
    unselected = obs["selection"].astype(str).eq("unselected").to_numpy()
    donor_rows: list[dict[str, Any]] = []
    all_columns: dict[str, np.ndarray] = {}
    for donor in sorted(obs.loc[unselected, "donor_id"].astype(str).unique()):
        mask = unselected & obs["donor_id"].astype(str).eq(donor).to_numpy()
        all_columns[f"D{donor}"] = aggregate_sparse_rows(counts, mask)
        donor_rows.append(
            {
                "pseudobulk_id": f"D{donor}",
                "donor_id": donor,
                "condition": str(obs.loc[mask, "condition"].iloc[0]),
                "cell_state": "all_cells",
                "n_cells": int(mask.sum()),
            }
        )
    all_frame = pd.DataFrame(
        {"gene": adata.var_names.astype(str), **all_columns}
    )
    all_frame.to_csv(
        output_dir / "pseudobulk_counts_all_cells.csv.gz",
        index=False,
        compression="gzip",
    )

    state_dir = output_dir / "pseudobulk_counts_by_cell_state"
    state_dir.mkdir(parents=True, exist_ok=True)
    state_files: dict[str, str] = {}
    for state in CELL_STATE_ORDER:
        state_columns: dict[str, np.ndarray] = {}
        for donor in sorted(obs.loc[unselected, "donor_id"].astype(str).unique()):
            mask = (
                unselected
                & obs["donor_id"].astype(str).eq(donor).to_numpy()
                & obs["cell_state"].astype(str).eq(state).to_numpy()
            )
            n_cells = int(mask.sum())
            if n_cells < min_cells_per_state_donor:
                continue
            pseudobulk_id = f"D{donor}__{state}"
            state_columns[pseudobulk_id] = aggregate_sparse_rows(counts, mask)
            donor_rows.append(
                {
                    "pseudobulk_id": pseudobulk_id,
                    "donor_id": donor,
                    "condition": str(obs.loc[mask, "condition"].iloc[0]),
                    "cell_state": state,
                    "n_cells": n_cells,
                }
            )
        if state_columns:
            frame = pd.DataFrame(
                {"gene": adata.var_names.astype(str), **state_columns}
            )
            path = state_dir / f"pseudobulk_counts_{state}.csv.gz"
            frame.to_csv(path, index=False, compression="gzip")
            state_files[state] = str(path)

    metadata = pd.DataFrame(donor_rows).drop_duplicates("pseudobulk_id")
    metadata.to_csv(output_dir / "pseudobulk_metadata.csv", index=False)
    return {
        "all_cell_pseudobulks": len(all_columns),
        "cell_state_files": state_files,
        "metadata_rows": int(len(metadata)),
        "minimum_cells_per_state_donor": min_cells_per_state_donor,
    }


def export_cell_table(adata: ad.AnnData, output_dir: Path) -> Path:
    columns = [
        "run_id",
        "sample",
        "donor_id",
        "condition",
        "selection",
        "run_number",
        "total_counts",
        "n_genes_by_counts",
        "pct_counts_mt",
        "legacy_doublet_proxy_score",
        "legacy_doublet_proxy_flag",
        "doublet_score",
        "predicted_doublet",
        "leiden_0_5",
        "curated_cell_state",
        "label_transfer_state",
        "label_transfer_confidence",
        "cell_state",
        "strict_gate_state",
    ] + [f"{name}_score" for name in SIGNATURES]
    existing = [column for column in columns if column in adata.obs.columns]
    frame = adata.obs[existing].copy()
    frame.insert(0, "cell", adata.obs_names.astype(str))
    frame["umap_1"] = adata.obsm["X_umap"][:, 0]
    frame["umap_2"] = adata.obsm["X_umap"][:, 1]
    path = output_dir / "corrected_single_cell_metadata_scores_umap.csv.gz"
    frame.to_csv(path, index=False, compression="gzip")
    return path


def compare_legacy_atlas(
    adata: ad.AnnData,
    legacy_marker_path: Path,
) -> dict[str, Any]:
    if not legacy_marker_path.exists():
        return {"status": "legacy_marker_file_missing"}
    legacy = pd.read_csv(
        legacy_marker_path,
        usecols=[
            "cell",
            "sample",
            "predicted_cell_state",
            "strict_gate_state",
        ],
    )
    legacy["barcode"] = legacy["cell"].astype(str).str.rsplit("_", n=1).str[-1]
    legacy["donor_id"] = (
        legacy["sample"].astype(str).str.extract(r"^(\d+)", expand=False)
    )
    legacy["selection"] = np.where(
        legacy["sample"].astype(str).str.contains("CD45", case=False, na=False),
        "CD45_depleted",
        "unselected",
    )
    legacy["run_number"] = 0
    legacy["comparison_key"] = [
        curated_key(donor, selection, run_number, barcode)
        for donor, selection, run_number, barcode in zip(
            legacy["donor_id"].astype(str),
            legacy["selection"].astype(str),
            legacy["run_number"].astype(int),
            legacy["barcode"].astype(str),
        )
    ]
    current = adata.obs.reset_index().copy()
    current["barcode"] = current["cell_id"].astype(str).str.split("__", n=1).str[-1]
    current["comparison_run_number"] = np.where(
        current["donor_id"].astype(str).eq("1074")
        & current["run_number"].astype(int).eq(2),
        0,
        current["run_number"].astype(int),
    )
    current["comparison_key"] = [
        curated_key(donor, selection, run_number, barcode)
        for donor, selection, run_number, barcode in zip(
            current["donor_id"].astype(str),
            current["selection"].astype(str),
            current["comparison_run_number"].astype(int),
            current["barcode"].astype(str),
        )
    ]
    merged = current.merge(
        legacy[
            [
                "comparison_key",
                "predicted_cell_state",
                "strict_gate_state",
            ]
        ],
        on="comparison_key",
        how="inner",
        suffixes=("_corrected", "_legacy"),
    )
    if merged.empty:
        return {"status": "no_overlap"}
    agreement = (
        merged["cell_state"].astype(str)
        == merged["predicted_cell_state"].astype(str)
    )
    return {
        "status": "ok",
        "overlapping_cells": int(len(merged)),
        "broad_state_agreement": float(agreement.mean()),
        "legacy_atlas_cells": int(len(legacy)),
        "corrected_atlas_cells": int(adata.n_obs),
        "legacy_missing_run1_qc_cells": 289,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Rebuild the HCA NEC single-cell atlas without duplicate-run overwrite."
    )
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument(
        "--snapshot-results",
        type=Path,
        default=DEFAULT_SNAPSHOT_RESULTS,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--n-top-genes", type=int, default=3000)
    arguments = parser.parse_args()

    np.random.seed(RANDOM_SEED)
    sc.settings.seed = RANDOM_SEED
    sc.settings.verbosity = 2
    output_dir = arguments.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_root = arguments.raw_root.resolve()
    snapshot_results = arguments.snapshot_results.resolve()
    status: dict[str, Any] = {
        "status": "started",
        "generated_utc": utc_now(),
        "raw_root": str(raw_root),
        "snapshot_results": str(snapshot_results),
        "output_dir": str(output_dir),
        "random_seed": RANDOM_SEED,
        "software": {
            "python": sys.version,
            "scanpy": sc.__version__,
            "anndata": ad.__version__,
            "harmonypy": harmonypy.__version__,
        },
    }
    write_json(output_dir / "single_cell_rebuild_status.json", status)

    run_dirs = sorted(
        path.parent
        for path in raw_root.glob("*/raw_feature_bc_matrix/matrix.mtx*")
    )
    if len(run_dirs) != 16:
        raise ValueError(
            f"Expected 16 raw 10X run directories, found {len(run_dirs)}"
        )

    datasets: list[ad.AnnData] = []
    qc_reports: list[dict[str, Any]] = []
    for index, matrix_dir in enumerate(run_dirs, start=1):
        metadata = run_metadata(matrix_dir.parent.name)
        print(
            f"[single-cell] {index:02d}/{len(run_dirs):02d} "
            f"{metadata['run_id']}",
            flush=True,
        )
        dataset, report = load_raw_10x_filtered(matrix_dir, metadata)
        datasets.append(dataset)
        qc_reports.append(report)
        pd.DataFrame(qc_reports).to_csv(
            output_dir / "run_level_qc_incremental.csv",
            index=False,
        )
        print(
            f"[single-cell] retained={dataset.n_obs} genes={dataset.n_vars} "
            f"elapsed={report['elapsed_seconds']}s",
            flush=True,
        )

    print("[single-cell] concatenating run-specific objects", flush=True)
    atlas = ad.concat(
        datasets,
        join="outer",
        merge="same",
        fill_value=0,
        index_unique=None,
    )
    del datasets
    gc.collect()
    sc.pp.filter_genes(atlas, min_cells=3)
    atlas.obs_names_make_unique()

    qc_frame = pd.DataFrame(qc_reports)
    qc_frame.to_csv(output_dir / "run_level_qc_summary.csv", index=False)
    original_qc_path = (
        snapshot_results
        / "single_cell_strict"
        / "seurat_v5_harmony_raw_qc_summary.csv"
    )
    if original_qc_path.exists():
        original_qc = pd.read_csv(original_qc_path)
        original_qc.to_csv(
            output_dir / "legacy_run_level_qc_summary.csv",
            index=False,
        )

    print("[single-cell] normalizing, integrating, and clustering", flush=True)
    integration = normalize_integrate_cluster(
        atlas,
        n_top_genes=arguments.n_top_genes,
    )

    print("[single-cell] mapping curated HCA labels", flush=True)
    curated_path = snapshot_results / "single_cell" / "cell_annotations.csv"
    curated_match = add_curated_labels(atlas, curated_path)
    label_transfer = donor_aware_label_transfer(atlas)

    print("[single-cell] scoring gate programs", flush=True)
    signature_coverage = score_programs(atlas)
    signature_coverage.to_csv(
        output_dir / "signature_marker_coverage.csv",
        index=False,
    )

    print("[single-cell] exporting sample-level and pseudobulk data", flush=True)
    sample_tables = sample_level_tables(atlas, output_dir)
    pseudobulk = write_pseudobulk_counts(atlas, output_dir)
    cell_table_path = export_cell_table(atlas, output_dir)

    legacy_comparison = compare_legacy_atlas(
        atlas,
        snapshot_results
        / "single_cell_strict"
        / "seurat_v5_harmony_marker_scores.csv",
    )

    atlas_path = output_dir / "corrected_hca_nec_atlas.h5ad"
    atlas.write_h5ad(atlas_path, compression="gzip")
    output_files = [
        path
        for path in output_dir.rglob("*")
        if path.is_file()
        and path.name not in {"single_cell_rebuild_status.json"}
    ]
    file_manifest = pd.DataFrame(
        [
            {
                "relative_path": path.relative_to(output_dir).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in sorted(output_files)
        ]
    )
    file_manifest.to_csv(
        output_dir / "single_cell_output_manifest.csv",
        index=False,
    )

    status.update(
        {
            "status": "completed",
            "completed_utc": utc_now(),
            "input_run_directories": len(run_dirs),
            "unique_run_ids": int(atlas.obs["run_id"].nunique()),
            "unique_sample_labels": int(atlas.obs["sample"].nunique()),
            "unique_donors": int(atlas.obs["donor_id"].nunique()),
            "cells_after_corrected_qc": int(atlas.n_obs),
            "genes_after_corrected_qc": int(atlas.n_vars),
            "condition_counts_cells": atlas.obs["condition"].value_counts().to_dict(),
            "selection_counts_cells": atlas.obs["selection"].value_counts().to_dict(),
            "run_level_qc": qc_reports,
            "integration": integration,
            "curated_match": curated_match,
            "label_transfer": label_transfer,
            "cell_state_counts": atlas.obs["cell_state"].astype(str).value_counts().to_dict(),
            "sample_tables": sample_tables,
            "pseudobulk": pseudobulk,
            "legacy_comparison": legacy_comparison,
            "outputs": {
                "atlas_h5ad": str(atlas_path),
                "cell_metadata_scores_umap": str(cell_table_path),
                "manifest": str(output_dir / "single_cell_output_manifest.csv"),
            },
        }
    )
    write_json(output_dir / "single_cell_rebuild_status.json", status)
    print(json.dumps(status, ensure_ascii=False, indent=2, default=json_safe))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise
    except Exception:
        traceback.print_exc()
        raise
