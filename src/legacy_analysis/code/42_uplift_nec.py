from __future__ import annotations

import gc
import json
from importlib import import_module

import anndata as ad
import numpy as np
import pandas as pd
import scvi
from scipy import sparse


common = import_module("40_story_chain_common")

LOCAL = common.LOCAL_REVISION_ROOT
ANALYSIS = common.ANALYSIS_ROOT
MODULE = "uplift_nec"
RNG = np.random.default_rng(common.SEED)
N_IMPUTATIONS = 250

STRICT_PATH = (
    LOCAL
    / "04_corrected_results"
    / "01_single_cell_corrected"
    / "corrected_hca_nec_atlas.h5ad"
)
DRCMAP_ROOT = ANALYSIS / "results" / "dr_cmap"

LINEAGES = {
    "epithelial": {
        "target": "M cell",
        "display": "M cell",
        "genes": [
            "SPIB",
            "GP2",
            "CCL20",
            "CXCL16",
            "TNFRSF11A",
            "SOX8",
            "IL22RA1",
            "ICAM1",
            "MUC2",
        ],
    },
    "endothelial": {
        "target": "LEC",
        "display": "Lymphatic endothelial cell",
        "genes": [
            "PROX1",
            "PDPN",
            "FLT4",
            "LYVE1",
            "CCL21",
            "ACKR4",
            "EFNA1",
        ],
    },
    "immune": {
        "target": "Treg",
        "display": "Regulatory T cell",
        "genes": [
            "FOXP3",
            "IL2RA",
            "CTLA4",
            "TIGIT",
            "CXCR6",
            "CCR7",
            "IL22",
        ],
    },
}


def save_csv_gzip(frame: pd.DataFrame, filename: str) -> None:
    for root in common.output_roots(MODULE):
        frame.to_csv(
            root / filename,
            index=False,
            encoding="utf-8",
            compression={"method": "gzip", "mtime": 0},
        )


def collect_full_posteriors() -> pd.DataFrame:
    frames = []
    for lineage, specification in LINEAGES.items():
        mapped_path = DRCMAP_ROOT / f"{lineage}_drcmap_mapped.h5ad"
        model_path = DRCMAP_ROOT / "models" / f"{lineage}_scanvi"
        mapped = ad.read_h5ad(mapped_path)
        query_mask = mapped.obs["query_kind"].eq("strict_nec_query")
        query = mapped[query_mask].copy()
        model = scvi.model.SCANVI.load(model_path, adata=mapped)
        probabilities = model.predict(
            adata=query,
            soft=True,
            batch_size=1024,
        )
        probabilities = probabilities.astype(np.float32)
        metadata = query.obs[
            [
                "cell_id_original",
                "donor_id",
                "sample_id",
                "condition",
                "selection",
                "scanvi_label",
            ]
        ].copy()
        metadata.index = probabilities.index
        probabilities.columns = [
            f"probability__{column}" for column in probabilities.columns
        ]
        frame = metadata.join(probabilities)
        probability_columns = [
            column for column in frame if column.startswith("probability__")
        ]
        matrix = frame[probability_columns].to_numpy(dtype=float)
        frame["posterior_max_probability"] = matrix.max(axis=1)
        frame["posterior_entropy_normalized"] = -np.sum(
            matrix * np.log(np.clip(matrix, 1e-12, None)),
            axis=1,
        ) / np.log(matrix.shape[1])
        frame["lineage_model"] = lineage
        frame["target_state"] = specification["target"]
        frames.append(frame.reset_index(drop=True))
        del model, mapped, query, probabilities
        gc.collect()
    posterior = pd.concat(frames, ignore_index=True)
    save_csv_gzip(posterior, "uplift_full_scanvi_posteriors.csv.gz")
    return posterior


def annotation_summary(posterior: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (lineage, condition, selection), group in posterior.groupby(
        ["lineage_model", "condition", "selection"],
        observed=True,
    ):
        rows.append(
            {
                "lineage_model": lineage,
                "condition": condition,
                "selection": selection,
                "cells": len(group),
                "median_max_probability": float(
                    group["posterior_max_probability"].median()
                ),
                "q25_max_probability": float(
                    group["posterior_max_probability"].quantile(0.25)
                ),
                "q75_max_probability": float(
                    group["posterior_max_probability"].quantile(0.75)
                ),
                "median_normalized_entropy": float(
                    group["posterior_entropy_normalized"].median()
                ),
                "fraction_max_probability_below_0_5": float(
                    group["posterior_max_probability"].lt(0.5).mean()
                ),
                "fraction_max_probability_below_0_7": float(
                    group["posterior_max_probability"].lt(0.7).mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def strict_primary_donor_table(strict: ad.AnnData) -> pd.DataFrame:
    obs = strict.obs.loc[strict.obs["selection"].eq("unselected")].copy()
    table = (
        obs.groupby(["donor_id", "condition"], observed=True)
        .size()
        .rename("total_unselected_cells")
        .reset_index()
    )
    table["donor_id"] = table["donor_id"].astype(str)
    return table


def abundance_inference(
    posterior: pd.DataFrame,
    donors: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    donor_rows = []
    effect_rows = []
    draw_rows = []
    for lineage, specification in LINEAGES.items():
        target = specification["target"]
        probability_column = f"probability__{target}"
        query = posterior.loc[
            posterior["lineage_model"].eq(lineage)
            & posterior["selection"].eq("unselected")
        ].copy()
        class_columns = [
            column
            for column in query
            if column.startswith("probability__") and query[column].notna().all()
        ]
        classes = [column.removeprefix("probability__") for column in class_columns]
        probabilities = query[class_columns].to_numpy(dtype=float)
        if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5):
            raise RuntimeError(f"Posterior probabilities do not sum to one for {lineage}")
        cumulative = np.cumsum(probabilities, axis=1)
        donor_ids = query["donor_id"].astype(str).to_numpy()
        hard_target = query["scanvi_label"].astype(str).eq(target).to_numpy()
        soft_target = query[probability_column].to_numpy(dtype=float)

        donor_values = donors.copy()
        hard_counts = {
            donor_id: int(hard_target[donor_ids == donor_id].sum())
            for donor_id in donor_values["donor_id"]
        }
        soft_counts = {
            donor_id: float(soft_target[donor_ids == donor_id].sum())
            for donor_id in donor_values["donor_id"]
        }
        donor_values["hard_count"] = donor_values["donor_id"].map(hard_counts)
        donor_values["posterior_expected_count"] = donor_values["donor_id"].map(
            soft_counts
        )
        donor_values["hard_fraction_all_cells"] = (
            donor_values["hard_count"] / donor_values["total_unselected_cells"]
        )
        donor_values["posterior_fraction_all_cells"] = (
            donor_values["posterior_expected_count"]
            / donor_values["total_unselected_cells"]
        )
        donor_values["posterior_count_variance"] = donor_values["donor_id"].map(
            {
                donor_id: float(
                    np.sum(
                        soft_target[donor_ids == donor_id]
                        * (1.0 - soft_target[donor_ids == donor_id])
                    )
                )
                for donor_id in donor_values["donor_id"]
            }
        )
        donor_values["lineage_model"] = lineage
        donor_values["target_state"] = target
        donor_rows.append(donor_values)

        draw_effects = []
        for draw in range(N_IMPUTATIONS):
            uniforms = RNG.random(len(query))
            assignments = np.sum(uniforms[:, None] > cumulative, axis=1)
            assigned_labels = np.asarray(classes, dtype=object)[assignments]
            target_mask = assigned_labels == target
            draw_values = donors.copy()
            draw_values["imputed_count"] = draw_values["donor_id"].map(
                {
                    donor_id: int(target_mask[donor_ids == donor_id].sum())
                    for donor_id in draw_values["donor_id"]
                }
            )
            draw_values["imputed_fraction_all_cells"] = (
                draw_values["imputed_count"]
                / draw_values["total_unselected_cells"]
            )
            draw_values["imputed_logit_fraction"] = np.log(
                (draw_values["imputed_count"] + 0.5)
                / (
                    draw_values["total_unselected_cells"]
                    - draw_values["imputed_count"]
                    + 0.5
                )
            )
            exact = common.exact_label_permutation(
                draw_values["imputed_logit_fraction"].to_numpy(dtype=float),
                draw_values["condition"].eq("nec").to_numpy(),
            )
            draw_effects.append(exact["effect"])
            draw_rows.extend(
                {
                    "lineage_model": lineage,
                    "target_state": target,
                    "imputation": draw + 1,
                    "donor_id": row.donor_id,
                    "condition": row.condition,
                    "imputed_count": row.imputed_count,
                    "total_unselected_cells": row.total_unselected_cells,
                    "imputed_fraction_all_cells": row.imputed_fraction_all_cells,
                }
                for row in draw_values.itertuples()
            )

        draw_effects_array = np.asarray(draw_effects, dtype=float)
        for method, count_column in [
            ("Hard scANVI label", "hard_count"),
            ("Posterior expectation", "posterior_expected_count"),
        ]:
            values = donor_values.copy()
            values["logit_fraction"] = np.log(
                (values[count_column] + 0.5)
                / (
                    values["total_unselected_cells"]
                    - values[count_column]
                    + 0.5
                )
            )
            exact = common.exact_label_permutation(
                values["logit_fraction"].to_numpy(dtype=float),
                values["condition"].eq("nec").to_numpy(),
            )
            effect_rows.append(
                {
                    "lineage_model": lineage,
                    "target_state": target,
                    "method": method,
                    "hedges_g_nec_vs_control": exact["effect"],
                    "exact_permutation_pvalue": exact["p_value"],
                    "exact_assignments": exact["assignments"],
                    "mi_effect_median": np.nan,
                    "mi_effect_q025": np.nan,
                    "mi_effect_q975": np.nan,
                    "mi_direction_probability": np.nan,
                    "mi_sign_tail_probability": np.nan,
                    "imputations": 0,
                }
            )
        median = float(np.nanmedian(draw_effects_array))
        mi_tail = float(
            min(
                1.0,
                2.0
                * min(
                    (1 + np.sum(draw_effects_array >= 0))
                    / (N_IMPUTATIONS + 1),
                    (1 + np.sum(draw_effects_array <= 0))
                    / (N_IMPUTATIONS + 1),
                ),
            )
        )
        effect_rows.append(
            {
                "lineage_model": lineage,
                "target_state": target,
                "method": "Posterior multiple imputation",
                "hedges_g_nec_vs_control": median,
                "exact_permutation_pvalue": np.nan,
                "exact_assignments": np.nan,
                "mi_effect_median": median,
                "mi_effect_q025": float(np.nanquantile(draw_effects_array, 0.025)),
                "mi_effect_q975": float(np.nanquantile(draw_effects_array, 0.975)),
                "mi_direction_probability": float(
                    max(
                        np.mean(draw_effects_array > 0),
                        np.mean(draw_effects_array < 0),
                    )
                ),
                "mi_sign_tail_probability": mi_tail,
                "imputations": N_IMPUTATIONS,
            }
        )
    donor_frame = pd.concat(donor_rows, ignore_index=True)
    effects = pd.DataFrame(effect_rows)
    effects["fdr_bh_hard_or_expected"] = np.nan
    eligible = effects["method"].isin(["Hard scANVI label", "Posterior expectation"])
    effects.loc[eligible, "fdr_bh_hard_or_expected"] = common.bh_fdr(
        effects.loc[eligible, "exact_permutation_pvalue"]
    )
    return donor_frame, effects, pd.DataFrame(draw_rows)


def weighted_logcpm(
    counts: np.ndarray,
    total_counts: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    library_size = float(np.sum(total_counts * weights))
    if library_size <= 0 or float(np.sum(weights)) <= 0:
        return np.full(counts.shape[1], np.nan, dtype=float)
    gene_counts = np.sum(counts * weights[:, None], axis=0)
    return np.log2((gene_counts + 0.5) / (library_size + 1.0) * 1_000_000.0)


def expression_inference(
    strict: ad.AnnData,
    posterior: pd.DataFrame,
    donors: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    strict_positions = pd.Series(
        np.arange(strict.n_obs),
        index=strict.obs_names.astype(str),
    )
    available_genes = set(strict.var_names.astype(str))
    donor_expression_rows = []
    summary_rows = []

    for lineage, specification in LINEAGES.items():
        target = specification["target"]
        query = posterior.loc[
            posterior["lineage_model"].eq(lineage)
            & posterior["selection"].eq("unselected")
        ].copy()
        genes = [gene for gene in specification["genes"] if gene in available_genes]
        query["strict_position"] = query["cell_id_original"].astype(str).map(
            strict_positions
        )
        if query["strict_position"].isna().any():
            raise RuntimeError(f"Unmatched strict cell identifiers in {lineage}")
        row_positions = query["strict_position"].astype(int).to_numpy()
        gene_positions = strict.var_names.get_indexer(genes)
        counts_layer = strict.layers["counts"]
        selected = counts_layer[row_positions][:, gene_positions]
        counts = selected.toarray() if sparse.issparse(selected) else np.asarray(selected)
        total_counts = strict.obs.iloc[row_positions]["total_counts"].to_numpy(dtype=float)
        donor_ids = query["donor_id"].astype(str).to_numpy()
        probability = query[f"probability__{target}"].to_numpy(dtype=float)
        hard = query["scanvi_label"].astype(str).eq(target).to_numpy(dtype=float)
        class_columns = [
            column
            for column in query
            if column.startswith("probability__") and query[column].notna().all()
        ]
        classes = np.asarray(
            [column.removeprefix("probability__") for column in class_columns],
            dtype=object,
        )
        cumulative = np.cumsum(query[class_columns].to_numpy(dtype=float), axis=1)
        if not np.allclose(cumulative[:, -1], 1.0, atol=1e-5):
            raise RuntimeError(f"Posterior probabilities do not sum to one for {lineage}")

        method_values: dict[str, list[dict[str, object]]] = {
            "Hard scANVI label": [],
            "Posterior expectation": [],
        }
        for donor in donors.itertuples():
            donor_mask = donor_ids == donor.donor_id
            for method, weights in [
                ("Hard scANVI label", hard[donor_mask]),
                ("Posterior expectation", probability[donor_mask]),
            ]:
                logcpm = weighted_logcpm(
                    counts[donor_mask],
                    total_counts[donor_mask],
                    weights,
                )
                for gene, value in zip(genes, logcpm, strict=True):
                    method_values[method].append(
                        {
                            "lineage_model": lineage,
                            "target_state": target,
                            "method": method,
                            "donor_id": donor.donor_id,
                            "condition": donor.condition,
                            "gene": gene,
                            "log2_cpm": float(value),
                            "effective_target_cells": float(np.sum(weights)),
                        }
                    )
        for rows in method_values.values():
            donor_expression_rows.extend(rows)

        mi_gene_effects = {gene: [] for gene in genes}
        for _ in range(N_IMPUTATIONS):
            uniforms = RNG.random(len(query))
            assignments = np.sum(uniforms[:, None] > cumulative, axis=1)
            target_weights = (classes[assignments] == target).astype(float)
            draw_rows = []
            for donor in donors.itertuples():
                donor_mask = donor_ids == donor.donor_id
                logcpm = weighted_logcpm(
                    counts[donor_mask],
                    total_counts[donor_mask],
                    target_weights[donor_mask],
                )
                draw_rows.extend(
                    {
                        "donor_id": donor.donor_id,
                        "condition": donor.condition,
                        "gene": gene,
                        "log2_cpm": float(value),
                    }
                    for gene, value in zip(genes, logcpm, strict=True)
                )
            draw_frame = pd.DataFrame(draw_rows)
            for gene, group in draw_frame.groupby("gene", sort=False):
                mi_gene_effects[gene].append(
                    common.hedges_g(
                        group.loc[group["condition"].eq("nec"), "log2_cpm"],
                        group.loc[group["condition"].eq("control"), "log2_cpm"],
                    )
                )

        expression_frame = pd.DataFrame(
            method_values["Hard scANVI label"]
            + method_values["Posterior expectation"]
        )
        for gene in genes:
            gene_values = np.asarray(mi_gene_effects[gene], dtype=float)
            for method in ["Hard scANVI label", "Posterior expectation"]:
                group = expression_frame.loc[
                    expression_frame["gene"].eq(gene)
                    & expression_frame["method"].eq(method)
                ]
                exact = common.exact_label_permutation(
                    group["log2_cpm"].to_numpy(dtype=float),
                    group["condition"].eq("nec").to_numpy(),
                )
                summary_rows.append(
                    {
                        "lineage_model": lineage,
                        "target_state": target,
                        "gene": gene,
                        "method": method,
                        "hedges_g_nec_vs_control": exact["effect"],
                        "exact_permutation_pvalue": exact["p_value"],
                        "exact_assignments": exact["assignments"],
                        "mi_effect_median": np.nan,
                        "mi_effect_q025": np.nan,
                        "mi_effect_q975": np.nan,
                        "mi_direction_probability": np.nan,
                        "mi_sign_tail_probability": np.nan,
                        "imputations": 0,
                    }
                )
            median = float(np.nanmedian(gene_values))
            finite_gene_values = gene_values[np.isfinite(gene_values)]
            if len(finite_gene_values) == 0:
                summary_rows.append(
                    {
                        "lineage_model": lineage,
                        "target_state": target,
                        "gene": gene,
                        "method": "Posterior multiple imputation",
                        "hedges_g_nec_vs_control": np.nan,
                        "exact_permutation_pvalue": np.nan,
                        "exact_assignments": np.nan,
                        "mi_effect_median": np.nan,
                        "mi_effect_q025": np.nan,
                        "mi_effect_q975": np.nan,
                        "mi_direction_probability": np.nan,
                        "mi_sign_tail_probability": np.nan,
                        "imputations": N_IMPUTATIONS,
                    }
                )
                continue
            mi_tail = float(
                min(
                    1.0,
                    2.0
                    * min(
                        (1 + np.sum(finite_gene_values >= 0))
                        / (len(finite_gene_values) + 1),
                        (1 + np.sum(finite_gene_values <= 0))
                        / (len(finite_gene_values) + 1),
                    ),
                )
            )
            summary_rows.append(
                {
                    "lineage_model": lineage,
                    "target_state": target,
                    "gene": gene,
                    "method": "Posterior multiple imputation",
                    "hedges_g_nec_vs_control": median,
                    "exact_permutation_pvalue": np.nan,
                    "exact_assignments": np.nan,
                    "mi_effect_median": median,
                    "mi_effect_q025": float(
                        np.nanquantile(finite_gene_values, 0.025)
                    ),
                    "mi_effect_q975": float(
                        np.nanquantile(finite_gene_values, 0.975)
                    ),
                    "mi_direction_probability": float(
                        max(
                            np.mean(finite_gene_values > 0),
                            np.mean(finite_gene_values < 0),
                        )
                    ),
                    "mi_sign_tail_probability": mi_tail,
                    "imputations": N_IMPUTATIONS,
                }
            )

    summary = pd.DataFrame(summary_rows)
    eligible = summary["method"].isin(
        ["Hard scANVI label", "Posterior expectation"]
    )
    summary["fdr_bh_within_method"] = np.nan
    summary.loc[eligible, "fdr_bh_within_method"] = summary.loc[
        eligible
    ].groupby("method", sort=False)["exact_permutation_pvalue"].transform(
        common.bh_fdr
    )
    return pd.DataFrame(donor_expression_rows), summary


def main() -> None:
    posterior = collect_full_posteriors()
    annotation = annotation_summary(posterior)
    strict = ad.read_h5ad(STRICT_PATH)
    donors = strict_primary_donor_table(strict)
    donor_abundance, abundance_effects, abundance_draws = abundance_inference(
        posterior,
        donors,
    )
    donor_expression, expression_effects = expression_inference(
        strict,
        posterior,
        donors,
    )

    common.write_csv(annotation, MODULE, "uplift_annotation_uncertainty_summary.csv")
    common.write_csv(donor_abundance, MODULE, "uplift_donor_abundance.csv")
    common.write_csv(abundance_effects, MODULE, "uplift_abundance_effects.csv")
    save_csv_gzip(abundance_draws, "uplift_abundance_imputation_draws.csv.gz")
    common.write_csv(
        donor_expression,
        MODULE,
        "uplift_target_gene_donor_logcpm.csv",
    )
    common.write_csv(
        expression_effects,
        MODULE,
        "uplift_target_gene_effects.csv",
    )

    summary = {
        "analysis": "UPLIFT-NEC",
        "status": "complete",
        "strict_query_cells": int(len(posterior)),
        "independent_unselected_donors": int(donors["donor_id"].nunique()),
        "lineage_models": list(LINEAGES),
        "target_states": [specification["target"] for specification in LINEAGES.values()],
        "posterior_classes": {
            lineage: [
                column.removeprefix("probability__")
                for column in posterior.loc[
                    posterior["lineage_model"].eq(lineage)
                ].columns
                if column.startswith("probability__")
                and posterior.loc[
                    posterior["lineage_model"].eq(lineage), column
                ].notna().any()
            ]
            for lineage in LINEAGES
        },
        "imputations": N_IMPUTATIONS,
        "abundance_effect_rows": int(len(abundance_effects)),
        "target_gene_effect_rows": int(len(expression_effects)),
        "minimum_hard_or_expected_abundance_fdr": float(
            abundance_effects["fdr_bh_hard_or_expected"].min()
        ),
        "input_sha256": {
            str(STRICT_PATH.relative_to(common.WORKSPACE_ROOT)): common.sha256(
                STRICT_PATH
            ),
            **{
                str(
                    (DRCMAP_ROOT / f"{lineage}_drcmap_mapped.h5ad").relative_to(
                        common.WORKSPACE_ROOT
                    )
                ): common.sha256(
                    DRCMAP_ROOT / f"{lineage}_drcmap_mapped.h5ad"
                )
                for lineage in LINEAGES
            },
        },
        "method_notes": [
            "Full native scANVI class posteriors were regenerated from saved models.",
            "All abundance denominators are unselected cells from the 11 independent donors.",
            "Target-state assignments were sampled 250 times from the complete class posterior.",
            "Target-gene effects use raw-count weighted pseudobulk log2 CPM.",
        ],
    }
    common.write_json(summary, MODULE, "uplift_nec_summary.json")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
