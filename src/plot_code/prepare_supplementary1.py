"""Reproduce atlas diagnostic display tables with the original definitions."""
import argparse, math
from pathlib import Path
import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse, stats

SINGLE_CELL_ROOT = None
STATS_ROOT = None

def read_csv(path):
    return pd.read_csv(path)

def hc3_regression(
    frame: pd.DataFrame,
    outcome: str,
    covariates: list[str],
) -> dict[str, float]:
    columns = [outcome, "condition", *covariates]
    work = frame[columns].dropna().copy()
    y = work[outcome].to_numpy(dtype=float)
    y = (y - y.mean()) / y.std(ddof=0)
    x_parts = [np.ones(len(work)), work["condition"].eq("nec").astype(float)]
    for covariate in covariates:
        values = work[covariate].to_numpy(dtype=float)
        values = (values - values.mean()) / values.std(ddof=0)
        x_parts.append(values)
    design = np.column_stack(x_parts)
    inverse = np.linalg.pinv(design.T @ design)
    beta = inverse @ design.T @ y
    residual = y - design @ beta
    leverage = np.einsum("ij,jk,ik->i", design, inverse, design)
    adjusted_residual = residual / np.clip(1.0 - leverage, 1e-8, None)
    meat = design.T @ np.diag(adjusted_residual**2) @ design
    covariance = inverse @ meat @ inverse
    standard_error = math.sqrt(max(covariance[1, 1], 0.0))
    degrees_freedom = max(len(work) - design.shape[1], 1)
    critical = stats.t.ppf(0.975, degrees_freedom)
    t_statistic = beta[1] / standard_error if standard_error > 0 else np.nan
    p_value = (
        2 * stats.t.sf(abs(t_statistic), degrees_freedom)
        if np.isfinite(t_statistic)
        else np.nan
    )
    return {
        "n_donors": len(work),
        "standardized_nec_beta": beta[1],
        "hc3_standard_error": standard_error,
        "ci_low": beta[1] - critical * standard_error,
        "ci_high": beta[1] + critical * standard_error,
        "p_value": p_value,
    }

def build_covariate_sensitivity(donor_table: pd.DataFrame) -> pd.DataFrame:
    host = read_csv(
        STATS_ROOT / "single_cell_host_state_map.csv"
    )
    clr = read_csv(STATS_ROOT / "single_cell_donor_composition_clr_wide.csv")
    outcomes = host.merge(clr, on=["donor_id", "condition"], how="left")
    outcomes = outcomes.merge(
        donor_table[
            [
                "donor_id",
                "gestational_age_weeks",
                "age_at_surgery_days",
            ]
        ],
        on="donor_id",
        how="left",
    )
    feature_map = {
        "Regulatory T cells CLR abundance": "T_reg",
        "T and NK cells CLR abundance": "T_NK",
        "M cells CLR abundance": "M_cell",
        "Lymphatic endothelial CLR abundance": "LEC",
        "Antigen sampling": "Antigen sampling",
        "Lymphatic support": "Lymphatic support",
        "Immune residency": "Immune residency",
        "Epithelial repair": "Epithelial repair",
        "Inflammatory injury": "Inflammatory injury",
        "Surveillance state": "surveillance_state",
        "Inflammatory injury state": "injury_state",
        "Injury-surveillance imbalance": "injury_surveillance_imbalance",
    }
    rows = []
    adjustments = {
        "Unadjusted": [],
        "Gestational age adjusted": ["gestational_age_weeks"],
        "Age at surgery adjusted": ["age_at_surgery_days"],
    }
    for label, column in feature_map.items():
        for adjustment, covariates in adjustments.items():
            result = hc3_regression(outcomes, column, covariates)
            rows.append(
                {
                    "feature": label,
                    "adjustment": adjustment,
                    **result,
                }
            )
    return pd.DataFrame(rows)

def build_capture_mode_pairs() -> pd.DataFrame:
    composition = read_csv(SINGLE_CELL_ROOT / "cell_state_proportions_capture_unit.csv")
    modules = read_csv(STATS_ROOT / "single_cell_module_scores_capture_unit.csv")
    paired_donors = [1056, 1127, 1212, 1214]
    rows = []
    selected_states = {
        "M_cell": "M cells",
        "LEC": "Lymphatic endothelial cells",
        "T_reg": "Regulatory T cells",
        "T_NK": "T and NK cells",
        "epithelial": "Epithelial cells",
        "myeloid": "Myeloid cells",
    }
    for state, label in selected_states.items():
        subset = composition.loc[
            composition["donor_id"].isin(paired_donors)
            & composition["cell_state"].eq(state)
        ]
        for row in subset.itertuples():
            rows.append(
                {
                    "donor_id": row.donor_id,
                    "feature_family": "Cell-state fraction",
                    "feature": label,
                    "capture_mode": row.selection,
                    "value": row.fraction,
                }
            )
    selected_modules = [
        "Antigen sampling",
        "Lymphatic support",
        "Immune residency",
        "Inflammatory injury",
    ]
    subset = modules.loc[
        modules["donor_id"].isin(paired_donors)
        & modules["module"].isin(selected_modules)
    ]
    for row in subset.itertuples():
        rows.append(
            {
                "donor_id": row.donor_id,
                "feature_family": "Gene-program score",
                "feature": row.module,
                "capture_mode": row.selection,
                "value": row.module_score,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["feature_family", "feature", "donor_id", "capture_mode"]
    )

MARKERS = {
    "M cells": ["SPIB", "GP2", "CCL20", "CXCL16"],
    "Lymphatic endothelial": ["PROX1", "PDPN", "FLT4", "LYVE1", "CCL21"],
    "Regulatory T cells": ["FOXP3", "IL2RA", "CTLA4", "TIGIT", "CXCR6", "CCR7"],
    "T and NK cells": ["CD3D", "NKG7", "GNLY"],
    "Epithelial": ["EPCAM", "KRT19", "MUC2", "IL22RA1", "EPHA2"],
    "Myeloid": ["LYZ", "S100A8", "S100A9", "TNF", "NLRP3"],
    "Vascular endothelial": ["PECAM1", "VWF", "ICAM1", "EFNA1"],
    "Fibroblast": ["COL1A1", "COL3A1", "VEGFC"],
    "Epithelial stem": ["LGR5", "OLFM4"],
}

STATE_LABELS = {
    "M_cell": "M cells",
    "LEC": "Lymphatic endothelial",
    "T_reg": "Regulatory T cells",
    "T_NK": "T and NK cells",
    "epithelial": "Epithelial",
    "myeloid": "Myeloid",
    "vascular_endothelial": "Vascular endothelial",
    "fibroblast": "Fibroblast",
    "epithelial_stem": "Epithelial stem",
}

def build_marker_dotplot() -> pd.DataFrame:
    h5ad_path = SINGLE_CELL_ROOT / "corrected_hca_nec_atlas.h5ad"
    all_markers = list(dict.fromkeys(gene for genes in MARKERS.values() for gene in genes))
    backed = ad.read_h5ad(h5ad_path, backed="r")
    present = [gene for gene in all_markers if gene in backed.var_names]
    atlas = backed[:, present].to_memory()
    expression = atlas.X
    counts = atlas.layers["counts"]
    if not sparse.issparse(expression):
        expression = sparse.csr_matrix(expression)
    if not sparse.issparse(counts):
        counts = sparse.csr_matrix(counts)
    rows = []
    state_order = list(STATE_LABELS)
    for state in state_order:
        mask = np.asarray(atlas.obs["cell_state"].eq(state))
        mean_expression = np.asarray(expression[mask].mean(axis=0)).ravel()
        fraction_expressing = (
            np.asarray((counts[mask] > 0).mean(axis=0)).ravel()
        )
        for gene, mean_value, fraction in zip(
            present, mean_expression, fraction_expressing, strict=True
        ):
            rows.append(
                {
                    "cell_state": STATE_LABELS[state],
                    "gene": gene,
                    "mean_log_normalized_expression": mean_value,
                    "fraction_expressing": fraction,
                }
            )
    frame = pd.DataFrame(rows)
    frame["mean_expression_z"] = frame.groupby("gene")[
        "mean_log_normalized_expression"
    ].transform(
        lambda values: (
            (values - values.mean()) / values.std(ddof=0)
            if values.std(ddof=0) > 0
            else 0.0
        )
    )
    frame["marker_family"] = frame["gene"].map(
        {
            gene: family
            for family, genes in MARKERS.items()
            for gene in genes
        }
    )
    return frame

def main():
    global SINGLE_CELL_ROOT, STATS_ROOT
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw-root',type=Path,required=True)
    p.add_argument('--stats-root',type=Path,required=True)
    p.add_argument('--donors',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args(); SINGLE_CELL_ROOT=a.raw_root; STATS_ROOT=a.stats_root
    a.out.mkdir(parents=True,exist_ok=True)
    donors=pd.read_csv(a.donors)
    donors['donor_id']=donors.donor_id.astype(str).str.extract(r'(\d+)',expand=False).astype(int)
    outputs={'marker_dotplot_data.csv':build_marker_dotplot(),
             'capture_mode_paired_values.csv':build_capture_mode_pairs(),
             'single_cell_covariate_sensitivity.csv':build_covariate_sensitivity(donors)}
    for name,frame in outputs.items():frame.to_csv(a.out/name,index=False)
    print({name:len(frame) for name,frame in outputs.items()},flush=True)

if __name__=='__main__':main()
