from __future__ import annotations

import json
import re
from importlib import import_module

import anndata as ad
import numpy as np
import ot
import pandas as pd
from scipy import sparse, stats
from scipy.spatial.distance import cdist


common = import_module("40_story_chain_common")

LOCAL = common.LOCAL_REVISION_ROOT
ANALYSIS = common.ANALYSIS_ROOT
MODULE = "gate_ot"
RNG = np.random.default_rng(common.SEED)

STRICT_PATH = (
    LOCAL
    / "04_corrected_results"
    / "01_single_cell_corrected"
    / "corrected_hca_nec_atlas.h5ad"
)
UPLIFT_ROOT = ANALYSIS / "results" / "story_chain_upgrade" / "uplift_nec"
ORGANOID_PATH = (
    ANALYSIS
    / "data"
    / "public"
    / "dr_cmap"
    / "GSE275771_P11N_M_cell_bulk_seq_NormalizedCounts.csv.gz"
)
ORGANOID_TREND_PATH = (
    ANALYSIS
    / "results"
    / "fxr_ferroptosis_ilc3"
    / "organoid_mcell_gene_trends.parquet"
)
MODULE_PATH = (
    LOCAL
    / "04_corrected_results"
    / "02_statistical_reanalysis"
    / "single_cell_module_marker_coverage.csv"
)

PRIMARY_REG = 0.05
PRIMARY_REG_M = 1.0
SELECTED_GENE_COUNT = 500
NULL_DRAWS = 1000


def stage_from_sample(sample: str) -> tuple[str, int]:
    if "_PPP" in sample:
        return "SPIB+ ICAM2+ GP2+", 3
    if "_PP" in sample:
        return "SPIB+ ICAM2+ GP2-", 2
    if "_P" in sample:
        return "SPIB+ ICAM2- GP2-", 1
    if "_N" in sample:
        return "SPIB-", 0
    raise ValueError(f"Unrecognized organoid sample: {sample}")


def weighted_mcell_pseudobulk() -> tuple[pd.DataFrame, pd.DataFrame]:
    posterior = pd.read_csv(
        UPLIFT_ROOT / "uplift_full_scanvi_posteriors.csv.gz"
    )
    posterior = posterior.loc[
        posterior["lineage_model"].eq("epithelial")
        & posterior["selection"].eq("unselected")
    ].copy()
    strict = ad.read_h5ad(STRICT_PATH)
    positions = pd.Series(
        np.arange(strict.n_obs),
        index=strict.obs_names.astype(str),
    )
    posterior["strict_position"] = posterior["cell_id_original"].astype(str).map(
        positions
    )
    if posterior["strict_position"].isna().any():
        raise RuntimeError("M-cell posterior identifiers do not align to strict atlas")
    row_positions = posterior["strict_position"].astype(int).to_numpy()
    counts = strict.layers["counts"][row_positions]
    if not sparse.issparse(counts):
        counts = sparse.csr_matrix(counts)
    else:
        counts = counts.tocsr()
    total_counts = strict.obs.iloc[row_positions]["total_counts"].to_numpy(dtype=float)
    donor_ids = posterior["donor_id"].astype(str).to_numpy()
    conditions = posterior["condition"].astype(str).to_numpy()
    posterior_weights = posterior["probability__M cell"].to_numpy(dtype=float)
    hard_weights = posterior["scanvi_label"].astype(str).eq("M cell").to_numpy(
        dtype=float
    )
    matrices = []
    metadata_rows = []
    for method, weights in [
        ("Posterior expectation", posterior_weights),
        ("Hard scANVI label", hard_weights),
    ]:
        donor_rows = []
        for donor_id in sorted(np.unique(donor_ids), key=int):
            mask = donor_ids == donor_id
            weighted_gene_counts = np.asarray(
                counts[mask].T @ weights[mask]
            ).ravel()
            weighted_library = float(np.sum(total_counts[mask] * weights[mask]))
            logcpm = np.log2(
                (weighted_gene_counts + 0.5)
                / (weighted_library + 1.0)
                * 1_000_000.0
            )
            donor_rows.append(logcpm)
            metadata_rows.append(
                {
                    "donor_id": donor_id,
                    "condition": str(conditions[mask][0]),
                    "weighting_method": method,
                    "effective_mcell_count": float(weights[mask].sum()),
                    "weighted_library_size": weighted_library,
                }
            )
        matrix = pd.DataFrame(
            donor_rows,
            index=[row["donor_id"] for row in metadata_rows[-len(donor_rows) :]],
            columns=strict.var_names.astype(str).str.upper(),
        )
        matrix.index.name = "donor_id"
        matrix["weighting_method"] = method
        matrices.append(matrix.reset_index())
    return pd.concat(matrices, ignore_index=True), pd.DataFrame(metadata_rows)


def prepare_organoid() -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.read_csv(ORGANOID_PATH, index_col=0)
    raw.index = raw.index.astype(str).str.upper()
    raw = raw.groupby(level=0).mean()
    expression = np.log2(raw.T + 1.0)
    metadata = pd.DataFrame({"sample_id": expression.index.astype(str)})
    stages = metadata["sample_id"].map(stage_from_sample)
    metadata["maturation_stage"] = [value[0] for value in stages]
    metadata["stage_index"] = [value[1] for value in stages]
    return expression, metadata


def excluded_gene(gene: str) -> bool:
    return bool(
        re.match(r"^(MT-|RPL\\d|RPS\\d)", gene)
        or gene in {"MALAT1", "NEAT1", "XIST"}
    )


def select_genes(
    donor_expression: pd.DataFrame,
    organoid_expression: pd.DataFrame,
) -> tuple[list[str], pd.DataFrame, list[str]]:
    trends = pd.read_parquet(ORGANOID_TREND_PATH)
    trends["gene"] = trends["gene"].astype(str).str.upper()
    common_genes = sorted(
        set(donor_expression.columns)
        & set(organoid_expression.columns)
        & set(trends["gene"])
    )
    common_genes = [gene for gene in common_genes if not excluded_gene(gene)]
    donor_variance = donor_expression[common_genes].var(axis=0, ddof=0)
    organoid_variance = organoid_expression[common_genes].var(axis=0, ddof=0)
    eligible = [
        gene
        for gene in common_genes
        if donor_variance[gene] > 0 and organoid_variance[gene] > 0
    ]
    trend_index = trends.drop_duplicates("gene").set_index("gene").loc[eligible]
    ranked = trend_index.sort_values(
        ["fdr_bh", "asymptotic_pvalue", "spearman_rho"],
        ascending=[True, True, False],
    )
    selected = ranked.head(SELECTED_GENE_COUNT).index.tolist()
    module_frame = pd.read_csv(MODULE_PATH)
    gate_genes = sorted(
        set(
            gene
            for values in module_frame["markers_available"].astype(str)
            for gene in values.upper().split(";")
        )
        & set(eligible)
    )
    selected = list(dict.fromkeys([*selected, *gate_genes]))
    inventory = ranked.reset_index()[
        ["gene", "slope_per_stage", "spearman_rho", "asymptotic_pvalue", "fdr_bh"]
    ].copy()
    inventory["selected_for_ot"] = inventory["gene"].isin(selected)
    inventory["gate_gene"] = inventory["gene"].isin(gate_genes)
    return selected, inventory, eligible


def zscore_by_gene(frame: pd.DataFrame, genes: list[str]) -> np.ndarray:
    matrix = frame[genes].to_numpy(dtype=float)
    mean = matrix.mean(axis=0, keepdims=True)
    standard_deviation = matrix.std(axis=0, ddof=0, keepdims=True)
    standard_deviation[standard_deviation == 0] = 1.0
    return (matrix - mean) / standard_deviation


def transport_map(
    donor_expression: pd.DataFrame,
    organoid_expression: pd.DataFrame,
    genes: list[str],
    reg: float,
    reg_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    donor_z = zscore_by_gene(donor_expression, genes)
    organoid_z = zscore_by_gene(organoid_expression, genes)
    cost = cdist(donor_z, organoid_z, metric="cosine")
    scale = float(np.median(cost[cost > 0])) if np.any(cost > 0) else 1.0
    cost = cost / scale
    donor_mass = np.repeat(1.0 / len(donor_z), len(donor_z))
    organoid_mass = np.repeat(1.0 / len(organoid_z), len(organoid_z))
    coupling = ot.unbalanced.sinkhorn_unbalanced(
        donor_mass,
        organoid_mass,
        cost,
        reg=reg,
        reg_m=reg_m,
        numItermax=5000,
        stopThr=1e-9,
    )
    row_sum = coupling.sum(axis=1, keepdims=True)
    probabilities = np.divide(
        coupling,
        row_sum,
        out=np.full_like(coupling, 1.0 / coupling.shape[1]),
        where=row_sum > 0,
    )
    return probabilities, cost


def summarize_transport(
    donor_ids: list[str],
    conditions: list[str],
    probabilities: np.ndarray,
    organoid_metadata: pd.DataFrame,
    weighting_method: str,
    reg: float,
    reg_m: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    stages = organoid_metadata["stage_index"].to_numpy(dtype=float)
    sample_rows = []
    donor_rows = []
    for index, donor_id in enumerate(donor_ids):
        probability = probabilities[index]
        barycentric_stage = float(np.sum(probability * stages))
        entropy = float(
            -np.sum(probability * np.log(np.clip(probability, 1e-12, None)))
            / np.log(len(probability))
        )
        donor_rows.append(
            {
                "donor_id": donor_id,
                "condition": conditions[index],
                "weighting_method": weighting_method,
                "reg": reg,
                "reg_m": reg_m,
                "barycentric_maturation_stage": barycentric_stage,
                "transport_entropy_normalized": entropy,
            }
        )
        for sample_index, sample in organoid_metadata.iterrows():
            sample_rows.append(
                {
                    "donor_id": donor_id,
                    "condition": conditions[index],
                    "weighting_method": weighting_method,
                    "reg": reg,
                    "reg_m": reg_m,
                    "organoid_sample": sample["sample_id"],
                    "maturation_stage": sample["maturation_stage"],
                    "stage_index": sample["stage_index"],
                    "transport_probability": float(probability[sample_index]),
                }
            )
    donor_frame = pd.DataFrame(donor_rows)
    exact = common.exact_label_permutation(
        donor_frame["barycentric_maturation_stage"].to_numpy(dtype=float),
        donor_frame["condition"].eq("nec").to_numpy(),
    )
    effect = {
        "hedges_g_nec_vs_control": exact["effect"],
        "exact_permutation_pvalue": exact["p_value"],
        "exact_assignments": exact["assignments"],
        "nec_mean_stage": float(
            donor_frame.loc[
                donor_frame["condition"].eq("nec"),
                "barycentric_maturation_stage",
            ].mean()
        ),
        "control_mean_stage": float(
            donor_frame.loc[
                donor_frame["condition"].eq("control"),
                "barycentric_maturation_stage",
            ].mean()
        ),
    }
    return donor_frame, pd.DataFrame(sample_rows), effect


def matched_null_test(
    donor_expression: pd.DataFrame,
    organoid_expression: pd.DataFrame,
    selected_genes: list[str],
    eligible_genes: list[str],
    organoid_metadata: pd.DataFrame,
    conditions: list[str],
) -> tuple[pd.DataFrame, dict[str, float]]:
    gene_mean = organoid_expression[eligible_genes].mean(axis=0)
    bins = pd.qcut(gene_mean, q=10, labels=False, duplicates="drop")
    genes_by_bin = {
        int(bin_id): list(group.index)
        for bin_id, group in gene_mean.groupby(bins)
    }
    selected_bins = bins.loc[selected_genes]
    selected_counts = selected_bins.value_counts().sort_index()
    stages = organoid_metadata["stage_index"].to_numpy(dtype=float)
    labels = np.asarray(conditions) == "nec"
    null_rows = []
    for draw in range(NULL_DRAWS):
        sampled = []
        for bin_id, count in selected_counts.items():
            candidates = [
                gene
                for gene in genes_by_bin[int(bin_id)]
                if gene not in selected_genes
            ]
            replace = len(candidates) < int(count)
            sampled.extend(
                RNG.choice(candidates, size=int(count), replace=replace).tolist()
            )
        probabilities, _ = transport_map(
            donor_expression,
            organoid_expression,
            sampled,
            PRIMARY_REG,
            PRIMARY_REG_M,
        )
        barycentric = probabilities @ stages
        effect = common.hedges_g(barycentric[labels], barycentric[~labels])
        null_rows.append(
            {
                "draw": draw + 1,
                "genes": len(sampled),
                "hedges_g_nec_vs_control": effect,
                "mean_stage_difference": float(
                    barycentric[labels].mean() - barycentric[~labels].mean()
                ),
            }
        )
    null = pd.DataFrame(null_rows)
    return null, {
        "null_draws": NULL_DRAWS,
        "matched_on": "organoid mean-expression decile",
    }


def gene_alignment(
    donor_expression: pd.DataFrame,
    conditions: list[str],
    selected_inventory: pd.DataFrame,
) -> pd.DataFrame:
    labels = np.asarray(conditions) == "nec"
    rows = []
    selected = selected_inventory.loc[selected_inventory["selected_for_ot"]]
    trend = selected.set_index("gene")
    for gene in selected["gene"]:
        values = donor_expression[gene].to_numpy(dtype=float)
        exact = common.exact_label_permutation(values, labels)
        organoid_slope = float(trend.loc[gene, "slope_per_stage"])
        rows.append(
            {
                "gene": gene,
                "donor_hedges_g_nec_vs_control": exact["effect"],
                "donor_exact_permutation_pvalue": exact["p_value"],
                "organoid_slope_per_stage": organoid_slope,
                "organoid_fdr_bh": float(trend.loc[gene, "fdr_bh"]),
                "maturation_discordance_score": float(
                    -exact["effect"] * organoid_slope
                ),
                "gate_gene": bool(trend.loc[gene, "gate_gene"]),
            }
        )
    frame = pd.DataFrame(rows)
    frame["donor_fdr_bh"] = common.bh_fdr(
        frame["donor_exact_permutation_pvalue"]
    )
    return frame.sort_values(
        "maturation_discordance_score",
        ascending=False,
    )


def main() -> None:
    donor_wide, donor_metadata = weighted_mcell_pseudobulk()
    organoid_expression, organoid_metadata = prepare_organoid()
    posterior_expression = donor_wide.loc[
        donor_wide["weighting_method"].eq("Posterior expectation")
    ].set_index("donor_id")
    posterior_expression = posterior_expression.drop(
        columns="weighting_method"
    )
    posterior_metadata = donor_metadata.loc[
        donor_metadata["weighting_method"].eq("Posterior expectation")
    ].copy()
    posterior_metadata = posterior_metadata.set_index("donor_id").loc[
        posterior_expression.index
    ]
    selected_genes, gene_inventory, eligible_genes = select_genes(
        posterior_expression,
        organoid_expression,
    )

    primary_probabilities, primary_cost = transport_map(
        posterior_expression,
        organoid_expression,
        selected_genes,
        PRIMARY_REG,
        PRIMARY_REG_M,
    )
    donor_stage, stage_probabilities, primary_effect = summarize_transport(
        posterior_expression.index.tolist(),
        posterior_metadata["condition"].astype(str).tolist(),
        primary_probabilities,
        organoid_metadata,
        "Posterior expectation",
        PRIMARY_REG,
        PRIMARY_REG_M,
    )

    sensitivity_rows = []
    for weighting_method in ["Posterior expectation", "Hard scANVI label"]:
        expression = donor_wide.loc[
            donor_wide["weighting_method"].eq(weighting_method)
        ].set_index("donor_id")
        expression = expression.drop(columns="weighting_method")
        metadata = donor_metadata.loc[
            donor_metadata["weighting_method"].eq(weighting_method)
        ].set_index("donor_id").loc[expression.index]
        for reg in [0.02, 0.05, 0.1, 0.2]:
            for reg_m in [0.1, 0.5, 1.0, 5.0]:
                probabilities, _ = transport_map(
                    expression,
                    organoid_expression,
                    selected_genes,
                    reg,
                    reg_m,
                )
                _, _, effect = summarize_transport(
                    expression.index.tolist(),
                    metadata["condition"].astype(str).tolist(),
                    probabilities,
                    organoid_metadata,
                    weighting_method,
                    reg,
                    reg_m,
                )
                sensitivity_rows.append(
                    {
                        "weighting_method": weighting_method,
                        "reg": reg,
                        "reg_m": reg_m,
                        **effect,
                    }
                )
    sensitivity = pd.DataFrame(sensitivity_rows)
    null, null_metadata = matched_null_test(
        posterior_expression,
        organoid_expression,
        selected_genes,
        eligible_genes,
        organoid_metadata,
        posterior_metadata["condition"].astype(str).tolist(),
    )
    observed_effect = primary_effect["hedges_g_nec_vs_control"]
    observed_difference = (
        primary_effect["nec_mean_stage"] - primary_effect["control_mean_stage"]
    )
    null_pvalue = float(
        (1 + np.sum(np.abs(null["hedges_g_nec_vs_control"]) >= abs(observed_effect)))
        / (len(null) + 1)
    )
    alignment = gene_alignment(
        posterior_expression,
        posterior_metadata["condition"].astype(str).tolist(),
        gene_inventory,
    )
    cost_frame = pd.DataFrame(
        primary_cost,
        index=posterior_expression.index,
        columns=organoid_expression.index,
    )
    cost_frame.index.name = "donor_id"
    cost_frame = cost_frame.reset_index()

    common.write_csv(donor_metadata, MODULE, "gate_ot_mcell_pseudobulk_metadata.csv")
    common.write_csv(donor_wide, MODULE, "gate_ot_mcell_pseudobulk_logcpm.csv")
    common.write_csv(gene_inventory, MODULE, "gate_ot_gene_inventory.csv")
    common.write_csv(organoid_metadata, MODULE, "gate_ot_organoid_metadata.csv")
    common.write_csv(donor_stage, MODULE, "gate_ot_donor_maturation_stage.csv")
    common.write_csv(
        stage_probabilities,
        MODULE,
        "gate_ot_donor_stage_probabilities.csv",
    )
    common.write_csv(cost_frame, MODULE, "gate_ot_primary_cost_matrix.csv")
    common.write_csv(sensitivity, MODULE, "gate_ot_hyperparameter_sensitivity.csv")
    common.write_csv(null, MODULE, "gate_ot_matched_null_distribution.csv")
    common.write_csv(alignment, MODULE, "gate_ot_gene_alignment.csv")

    summary = {
        "analysis": "GATE-OT",
        "status": "complete",
        "independent_donors": int(len(posterior_expression)),
        "organoid_samples": int(len(organoid_expression)),
        "ordered_organoid_stages": int(
            organoid_metadata["stage_index"].nunique()
        ),
        "selected_genes": int(len(selected_genes)),
        "eligible_common_genes": int(len(eligible_genes)),
        "primary_reg": PRIMARY_REG,
        "primary_reg_m": PRIMARY_REG_M,
        **primary_effect,
        "mean_stage_difference_nec_minus_control": observed_difference,
        "matched_null_two_sided_pvalue": null_pvalue,
        "hyperparameter_direction_stability": float(
            np.mean(
                np.sign(sensitivity["hedges_g_nec_vs_control"])
                == np.sign(observed_effect)
            )
        ),
        "posterior_only_hyperparameter_direction_stability": float(
            np.mean(
                np.sign(
                    sensitivity.loc[
                        sensitivity["weighting_method"].eq(
                            "Posterior expectation"
                        ),
                        "hedges_g_nec_vs_control",
                    ]
                )
                == np.sign(observed_effect)
            )
        ),
        "pot_version": ot.__version__,
        "null_test": null_metadata,
        "method_notes": [
            "M-cell donor profiles are full-transcriptome raw-count pseudobulks weighted by native scANVI posterior probabilities.",
            "The organoid reference contains eight samples spanning four ordered SPIB, ICAM2 and GP2 maturation stages.",
            "Within-cohort gene standardization precedes cosine-cost unbalanced entropic optimal transport.",
            "Diagnosis labels are not used to estimate the transport map.",
            "Inference uses exhaustive donor-label permutation and 1,000 expression-decile-matched null gene sets.",
        ],
    }
    common.write_json(summary, MODULE, "gate_ot_summary.json")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
