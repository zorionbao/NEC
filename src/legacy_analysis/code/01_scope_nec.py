from __future__ import annotations

import hashlib
import itertools
import json
import math
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


SEED = 20260726
RNG = np.random.default_rng(SEED)
ANALYSIS_ROOT = Path(__file__).resolve().parent.parent
PACKAGE_ROOT = ANALYSIS_ROOT.parent
WORKSPACE_ROOT = ANALYSIS_ROOT.parent
SNAPSHOT_ROOT = (
    WORKSPACE_ROOT
    / "."
    / "04_corrected_results"
    / "00_remote_snapshot"
)
SOURCE_ROOT = (
    SNAPSHOT_ROOT
    / "results"
    / "microbiome_strict"
)
SOURCE_HUMANN = SOURCE_ROOT / "humann_native_subset" / "samples"
SOURCE_MANIFEST = SOURCE_ROOT / "nec_metatranscriptome_full_run_manifest.csv"

SYNC_ROOT = ANALYSIS_ROOT / "data" / "synced" / "scope_humann"
RESULT_ROOT = ANALYSIS_ROOT / "results" / "scope_nec"
SOURCE_DATA_ROOT = (
    ANALYSIS_ROOT
    / "source_data"
    / "Extended_Data_Figure_5_SCOPE_NEC"
)
MANIFEST_ROOT = ANALYSIS_ROOT / "manifests"

EXCLUDED_PATHWAYS = {"UNMAPPED", "UNINTEGRATED"}
TARGET_PATTERN = re.compile(
    r"tricarbox|tca|glyox|pyruvate|propanoate|propionat|succinat",
    flags=re.IGNORECASE,
)


def ensure_local_write(path: Path) -> Path:
    resolved = path.resolve()
    if ANALYSIS_ROOT.resolve() not in resolved.parents and resolved != ANALYSIS_ROOT.resolve():
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
    out = np.full(values.shape, np.nan)
    valid = np.isfinite(values)
    if not valid.any():
        return out
    p = values[valid]
    order = np.argsort(p)
    ranked = p[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    restored = np.empty_like(adjusted)
    restored[order] = adjusted
    out[valid] = restored
    return out


def hedges_g_matrix(x: np.ndarray, case_mask: np.ndarray) -> np.ndarray:
    cases = x[:, case_mask]
    controls = x[:, ~case_mask]
    n1 = cases.shape[1]
    n0 = controls.shape[1]
    mean1 = np.mean(cases, axis=1)
    mean0 = np.mean(controls, axis=1)
    var1 = np.var(cases, axis=1, ddof=1)
    var0 = np.var(controls, axis=1, ddof=1)
    pooled = np.sqrt(
        np.maximum(
            ((n1 - 1) * var1 + (n0 - 1) * var0) / (n1 + n0 - 2),
            0.0,
        )
    )
    d = np.divide(
        mean1 - mean0,
        pooled,
        out=np.zeros_like(mean1),
        where=pooled > 0,
    )
    correction = 1 - 3 / (4 * (n1 + n0) - 9)
    return d * correction


def exact_group_permutation_p(
    x: np.ndarray,
    case_mask: np.ndarray,
) -> np.ndarray:
    n_samples = x.shape[1]
    n_case = int(case_mask.sum())
    total = np.sum(x, axis=1)
    observed = np.mean(x[:, case_mask], axis=1) - np.mean(
        x[:, ~case_mask],
        axis=1,
    )
    exceed = np.zeros(x.shape[0], dtype=np.int64)
    n_perm = 0
    for case_index in itertools.combinations(range(n_samples), n_case):
        case_index = np.asarray(case_index, dtype=int)
        case_sum = np.sum(x[:, case_index], axis=1)
        diff = case_sum / n_case - (total - case_sum) / (n_samples - n_case)
        exceed += np.abs(diff) >= (np.abs(observed) - 1e-12)
        n_perm += 1
    return exceed / n_perm


def exact_sign_flip_p(changes: np.ndarray) -> np.ndarray:
    observed = np.mean(changes, axis=1)
    exceed = np.zeros(changes.shape[0], dtype=np.int64)
    patterns = list(itertools.product((-1.0, 1.0), repeat=changes.shape[1]))
    for pattern in patterns:
        flipped = changes * np.asarray(pattern)[None, :]
        diff = np.mean(flipped, axis=1)
        exceed += np.abs(diff) >= (np.abs(observed) - 1e-12)
    return exceed / len(patterns)


def loo_sign_stability(x: np.ndarray, case_mask: np.ndarray) -> np.ndarray:
    full = np.mean(x[:, case_mask], axis=1) - np.mean(
        x[:, ~case_mask],
        axis=1,
    )
    concordant = np.zeros(x.shape[0], dtype=float)
    valid = np.zeros(x.shape[0], dtype=float)
    for held_out in range(x.shape[1]):
        keep = np.ones(x.shape[1], dtype=bool)
        keep[held_out] = False
        local_case = case_mask[keep]
        if local_case.sum() == 0 or (~local_case).sum() == 0:
            continue
        local = np.mean(x[:, keep][:, local_case], axis=1) - np.mean(
            x[:, keep][:, ~local_case],
            axis=1,
        )
        informative = (np.abs(full) > 1e-12) & (np.abs(local) > 1e-12)
        concordant[informative] += (
            np.sign(full[informative]) == np.sign(local[informative])
        )
        valid[informative] += 1
    return np.divide(
        concordant,
        valid,
        out=np.zeros_like(concordant),
        where=valid > 0,
    )


def bootstrap_group_difference(
    x: np.ndarray,
    case_mask: np.ndarray,
    n_boot: int = 5000,
) -> tuple[np.ndarray, np.ndarray]:
    cases = x[:, case_mask]
    controls = x[:, ~case_mask]
    boot = np.empty((x.shape[0], n_boot), dtype=np.float32)
    for index in range(n_boot):
        case_draw = RNG.integers(0, cases.shape[1], size=cases.shape[1])
        control_draw = RNG.integers(0, controls.shape[1], size=controls.shape[1])
        boot[:, index] = (
            np.mean(cases[:, case_draw], axis=1)
            - np.mean(controls[:, control_draw], axis=1)
        )
    return (
        np.quantile(boot, 0.025, axis=1),
        np.quantile(boot, 0.975, axis=1),
    )


def clean_taxon(value: str) -> str:
    if value.lower() == "unclassified":
        return "Unclassified contribution"
    species_match = re.search(r"s__([^|]+)$", value)
    if species_match:
        return species_match.group(1).replace("_", " ")
    genus_match = re.search(r"g__([^|]+)$", value)
    if genus_match:
        return genus_match.group(1).replace("_", " ")
    return value.replace("_", " ")


def split_pathway(value: str) -> tuple[str, str]:
    if ": " in value:
        pathway_id, pathway_name = value.split(": ", 1)
        return pathway_id, pathway_name
    return value, value


def parse_run_label(label: str) -> tuple[str, str, str]:
    acute = re.fullmatch(r"NEC-D(\d+)", label)
    recovery = re.fullmatch(r"NEC-R(\d+)", label)
    control = re.fullmatch(r"NOR(\d+)", label)
    if acute:
        return f"NEC_{acute.group(1)}", "NEC", "acute"
    if recovery:
        return f"NEC_{recovery.group(1)}", "NEC", "recovery"
    if control:
        return f"Control_{control.group(1)}", "Control", "control"
    raise ValueError(f"Unrecognized run label: {label}")


def synchronize_inputs() -> tuple[pd.DataFrame, list[Path], pd.DataFrame]:
    for directory in (SYNC_ROOT, RESULT_ROOT, SOURCE_DATA_ROOT, MANIFEST_ROOT):
        ensure_local_write(directory).mkdir(parents=True, exist_ok=True)
    if not SOURCE_MANIFEST.exists():
        raise FileNotFoundError(SOURCE_MANIFEST)
    source_files = sorted(SOURCE_HUMANN.glob("*/*_pathabundance.tsv"))
    if len(source_files) != 19:
        raise RuntimeError(f"Expected 19 HUMAnN files, found {len(source_files)}")

    inventory = []
    local_files = []
    for source in [SOURCE_MANIFEST, *source_files]:
        destination = SYNC_ROOT / source.name
        source_hash = sha256(source)
        if not destination.exists() or sha256(destination) != source_hash:
            shutil.copy2(source, destination)
        local_files.append(destination)
        inventory.append(
            {
                "source_file": str(source),
                "local_file": str(destination.relative_to(ANALYSIS_ROOT)),
                "bytes": destination.stat().st_size,
                "sha256": source_hash,
            }
        )
    inventory_frame = pd.DataFrame(inventory)
    inventory_frame.to_csv(
        ensure_local_write(MANIFEST_ROOT / "scope_input_inventory.csv"),
        index=False,
    )
    manifest = pd.read_csv(SYNC_ROOT / SOURCE_MANIFEST.name)
    manifest["label"] = manifest["isolate"].fillna(manifest["title"])
    parsed = manifest["label"].map(parse_run_label)
    manifest[["infant_id", "analysis_condition", "phase"]] = pd.DataFrame(
        parsed.tolist(),
        index=manifest.index,
    )
    return manifest, local_files[1:], inventory_frame


def parse_humann(
    manifest: pd.DataFrame,
    local_files: list[Path],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metadata = manifest.set_index("run")[
        ["label", "infant_id", "analysis_condition", "phase"]
    ].to_dict("index")
    total_rows: list[dict[str, object]] = []
    contribution_rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []

    for file_path in local_files:
        run = file_path.name.replace("_pathabundance.tsv", "")
        if run not in metadata:
            raise RuntimeError(f"Run missing from manifest: {run}")
        frame = pd.read_csv(
            file_path,
            sep="\t",
            comment="#",
            header=None,
            names=["feature", "abundance_rpk"],
        )
        frame["abundance_rpk"] = pd.to_numeric(
            frame["abundance_rpk"],
            errors="raise",
        )
        unstratified = frame[
            ~frame["feature"].str.contains(r"\|", regex=True)
        ].copy()
        unstratified = unstratified[
            ~unstratified["feature"].isin(EXCLUDED_PATHWAYS)
        ]
        stratified = frame[
            frame["feature"].str.contains(r"\|", regex=True)
        ].copy()
        stratified[["pathway", "taxon_raw"]] = stratified[
            "feature"
        ].str.split("|", n=1, expand=True)
        stratified = stratified[
            ~stratified["pathway"].isin(EXCLUDED_PATHWAYS)
        ]
        denominator = float(unstratified["abundance_rpk"].sum())
        if denominator <= 0:
            raise RuntimeError(f"Non-positive pathway denominator: {run}")
        scale = 1_000_000.0 / denominator

        scaled_pathways = 0
        residual_shares = []
        closure_errors = []
        for pathway, total_rpk in unstratified[
            ["feature", "abundance_rpk"]
        ].itertuples(index=False):
            pathway_id, pathway_name = split_pathway(pathway)
            subset = stratified[stratified["pathway"] == pathway]
            contributions = (
                subset.groupby("taxon_raw", observed=True)["abundance_rpk"]
                .sum()
                .to_dict()
            )
            explicit_sum = float(sum(contributions.values()))
            adjustment_factor = 1.0
            if explicit_sum > total_rpk and explicit_sum > 0:
                adjustment_factor = float(total_rpk / explicit_sum)
                contributions = {
                    key: value * adjustment_factor
                    for key, value in contributions.items()
                }
                explicit_sum = float(sum(contributions.values()))
                scaled_pathways += 1
            residual = max(float(total_rpk - explicit_sum), 0.0)
            cleaned: dict[str, float] = {}
            for taxon_raw, value in contributions.items():
                taxon = clean_taxon(taxon_raw)
                cleaned[taxon] = cleaned.get(taxon, 0.0) + float(value)
            if residual > 0:
                key = "Unclassified contribution"
                cleaned[key] = cleaned.get(key, 0.0) + residual
            if not cleaned and total_rpk > 0:
                cleaned["Unclassified contribution"] = float(total_rpk)

            reconstructed = float(sum(cleaned.values()))
            closure_error = abs(reconstructed - total_rpk)
            closure_errors.append(closure_error)
            residual_shares.append(
                cleaned.get("Unclassified contribution", 0.0)
                / total_rpk
                if total_rpk > 0
                else 0.0
            )
            common = {
                "run": run,
                **metadata[run],
                "pathway": pathway,
                "pathway_id": pathway_id,
                "pathway_name": pathway_name,
            }
            total_cpm = float(total_rpk * scale)
            total_rows.append(
                {
                    **common,
                    "total_pathway_cpm": total_cpm,
                }
            )
            for taxon, contribution_rpk in cleaned.items():
                contribution_cpm = float(contribution_rpk * scale)
                contribution_rows.append(
                    {
                        **common,
                        "taxon": taxon,
                        "contribution_cpm": contribution_cpm,
                        "total_pathway_cpm": total_cpm,
                        "carrier_share": (
                            contribution_rpk / total_rpk
                            if total_rpk > 0
                            else 0.0
                        ),
                        "adjustment_factor": adjustment_factor,
                    }
                )
        audit_rows.append(
            {
                "run": run,
                **metadata[run],
                "eligible_pathways": len(unstratified),
                "scaled_pathways": scaled_pathways,
                "median_unclassified_share": float(
                    np.median(residual_shares)
                ),
                "maximum_closure_error_rpk": float(max(closure_errors, default=0)),
            }
        )

    totals = pd.DataFrame(total_rows)
    contributions = pd.DataFrame(contribution_rows)
    audit = pd.DataFrame(audit_rows)
    if audit["maximum_closure_error_rpk"].max() > 1e-6:
        raise RuntimeError("SCOPE closure invariant failed")
    return totals, contributions, audit


def build_infant_matrices(
    totals: pd.DataFrame,
    contributions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], np.ndarray]:
    infants = (
        totals[["infant_id", "analysis_condition"]]
        .drop_duplicates()
        .sort_values(["analysis_condition", "infant_id"])
    )
    infant_order = infants["infant_id"].tolist()
    case_mask = (
        infants.set_index("infant_id")
        .loc[infant_order, "analysis_condition"]
        .eq("NEC")
        .to_numpy()
    )
    run_metadata = (
        totals[["run", "infant_id", "analysis_condition"]]
        .drop_duplicates()
        .sort_values("run")
    )
    pathway_features = totals[
        ["pathway", "pathway_id", "pathway_name"]
    ].drop_duplicates()
    complete_totals = (
        run_metadata.assign(_join_key=1)
        .merge(pathway_features.assign(_join_key=1), on="_join_key")
        .drop(columns="_join_key")
        .merge(
            totals[
                [
                    "run",
                    "pathway",
                    "total_pathway_cpm",
                ]
            ],
            on=["run", "pathway"],
            how="left",
            validate="one_to_one",
        )
    )
    complete_totals["total_pathway_cpm"] = complete_totals[
        "total_pathway_cpm"
    ].fillna(0.0)
    infant_totals = (
        complete_totals.groupby(
            ["infant_id", "analysis_condition", "pathway", "pathway_id", "pathway_name"],
            observed=True,
        )["total_pathway_cpm"]
        .mean()
        .reset_index()
    )
    contribution_features = contributions[
        ["pathway", "pathway_id", "pathway_name", "taxon"]
    ].drop_duplicates()
    complete_contributions = (
        run_metadata.assign(_join_key=1)
        .merge(contribution_features.assign(_join_key=1), on="_join_key")
        .drop(columns="_join_key")
        .merge(
            contributions[
                [
                    "run",
                    "pathway",
                    "taxon",
                    "contribution_cpm",
                ]
            ],
            on=["run", "pathway", "taxon"],
            how="left",
            validate="one_to_one",
        )
    )
    complete_contributions["contribution_cpm"] = complete_contributions[
        "contribution_cpm"
    ].fillna(0.0)
    infant_contributions = (
        complete_contributions.groupby(
            [
                "infant_id",
                "analysis_condition",
                "pathway",
                "pathway_id",
                "pathway_name",
                "taxon",
            ],
            observed=True,
        )["contribution_cpm"]
        .mean()
        .reset_index()
    )
    return infant_totals, infant_contributions, infant_order, case_mask


def pathway_statistics(
    infant_totals: pd.DataFrame,
    infant_order: list[str],
    case_mask: np.ndarray,
) -> pd.DataFrame:
    pivot = infant_totals.pivot_table(
        index=["pathway", "pathway_id", "pathway_name"],
        columns="infant_id",
        values="total_pathway_cpm",
        fill_value=0.0,
    ).reindex(columns=infant_order, fill_value=0.0)
    x = np.log1p(pivot.to_numpy(dtype=float))
    raw = pivot.to_numpy(dtype=float)
    effect = hedges_g_matrix(x, case_mask)
    pvalue = exact_group_permutation_p(x, case_mask)
    delta_log = np.mean(x[:, case_mask], axis=1) - np.mean(
        x[:, ~case_mask],
        axis=1,
    )
    ci_low, ci_high = bootstrap_group_difference(x, case_mask)
    result = pivot.index.to_frame(index=False)
    result["nec_mean_cpm"] = np.mean(raw[:, case_mask], axis=1)
    result["control_mean_cpm"] = np.mean(raw[:, ~case_mask], axis=1)
    result["delta_log1p_cpm"] = delta_log
    result["delta_log1p_cpm_ci_low"] = ci_low
    result["delta_log1p_cpm_ci_high"] = ci_high
    result["hedges_g"] = effect
    result["permutation_p"] = pvalue
    result["fdr"] = bh_adjust(pvalue)
    result["loo_sign_stability"] = loo_sign_stability(x, case_mask)
    result["prevalent_infants"] = np.sum(raw > 0, axis=1)
    result["target_carbon_pathway"] = result["pathway"].str.contains(
        TARGET_PATTERN,
        regex=True,
    )
    return result


def contribution_statistics(
    infant_contributions: pd.DataFrame,
    infant_order: list[str],
    case_mask: np.ndarray,
) -> pd.DataFrame:
    pivot = infant_contributions.pivot_table(
        index=["pathway", "pathway_id", "pathway_name", "taxon"],
        columns="infant_id",
        values="contribution_cpm",
        fill_value=0.0,
    ).reindex(columns=infant_order, fill_value=0.0)
    raw = pivot.to_numpy(dtype=float)
    x = np.log1p(raw)
    delta_cpm = np.mean(raw[:, case_mask], axis=1) - np.mean(
        raw[:, ~case_mask],
        axis=1,
    )
    delta_log = np.mean(x[:, case_mask], axis=1) - np.mean(
        x[:, ~case_mask],
        axis=1,
    )
    result = pivot.index.to_frame(index=False)
    result["nec_mean_contribution_cpm"] = np.mean(raw[:, case_mask], axis=1)
    result["control_mean_contribution_cpm"] = np.mean(
        raw[:, ~case_mask],
        axis=1,
    )
    result["delta_contribution_cpm"] = delta_cpm
    result["delta_log1p_contribution_cpm"] = delta_log
    result["hedges_g"] = hedges_g_matrix(x, case_mask)
    result["permutation_p"] = exact_group_permutation_p(x, case_mask)
    result["fdr"] = bh_adjust(result["permutation_p"].to_numpy())
    result["loo_sign_stability"] = loo_sign_stability(x, case_mask)
    result["prevalent_infants"] = np.sum(raw > 0, axis=1)
    denominator = result.groupby("pathway", observed=True)[
        "delta_contribution_cpm"
    ].transform(lambda values: np.sum(np.abs(values)))
    result["signed_attribution"] = np.divide(
        result["delta_contribution_cpm"],
        denominator,
        out=np.zeros(len(result), dtype=float),
        where=denominator.to_numpy() > 0,
    )
    result["target_carbon_pathway"] = result["pathway"].str.contains(
        TARGET_PATTERN,
        regex=True,
    )
    return result


def paired_statistics(
    totals: pd.DataFrame,
    contributions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    paired_runs = []
    for infant_number in range(1, 7):
        infant = f"NEC_{infant_number}"
        phases = totals[totals["infant_id"] == infant]["phase"].unique()
        if not {"acute", "recovery"}.issubset(set(phases)):
            raise RuntimeError(f"Missing paired phases for {infant}: {phases}")
        paired_runs.append(infant)

    total_pivot = totals[totals["infant_id"].isin(paired_runs)].pivot_table(
        index=["pathway", "pathway_id", "pathway_name"],
        columns=["infant_id", "phase"],
        values="total_pathway_cpm",
        fill_value=0.0,
    )
    total_changes = []
    for infant in paired_runs:
        total_changes.append(
            np.log1p(total_pivot[(infant, "recovery")].to_numpy())
            - np.log1p(total_pivot[(infant, "acute")].to_numpy())
        )
    total_change_matrix = np.column_stack(total_changes)
    total_result = total_pivot.index.to_frame(index=False)
    total_result["mean_recovery_minus_acute_log1p_cpm"] = np.mean(
        total_change_matrix,
        axis=1,
    )
    total_result["median_recovery_minus_acute_log1p_cpm"] = np.median(
        total_change_matrix,
        axis=1,
    )
    total_result["paired_sign_concordance"] = np.mean(
        np.sign(total_change_matrix)
        == np.sign(np.mean(total_change_matrix, axis=1))[:, None],
        axis=1,
    )
    total_result["sign_flip_p"] = exact_sign_flip_p(total_change_matrix)
    total_result["fdr"] = bh_adjust(total_result["sign_flip_p"].to_numpy())

    contribution_pivot = contributions[
        contributions["infant_id"].isin(paired_runs)
    ].pivot_table(
        index=["pathway", "pathway_id", "pathway_name", "taxon"],
        columns=["infant_id", "phase"],
        values="contribution_cpm",
        fill_value=0.0,
    )
    contribution_changes = []
    for infant in paired_runs:
        contribution_changes.append(
            np.log1p(
                contribution_pivot.get(
                    (infant, "recovery"),
                    pd.Series(
                        0.0,
                        index=contribution_pivot.index,
                    ),
                ).to_numpy()
            )
            - np.log1p(
                contribution_pivot.get(
                    (infant, "acute"),
                    pd.Series(
                        0.0,
                        index=contribution_pivot.index,
                    ),
                ).to_numpy()
            )
        )
    contribution_change_matrix = np.column_stack(contribution_changes)
    contribution_result = contribution_pivot.index.to_frame(index=False)
    contribution_result["mean_recovery_minus_acute_log1p_cpm"] = np.mean(
        contribution_change_matrix,
        axis=1,
    )
    contribution_result["paired_sign_concordance"] = np.mean(
        np.sign(contribution_change_matrix)
        == np.sign(
            np.mean(contribution_change_matrix, axis=1)
        )[:, None],
        axis=1,
    )
    contribution_result["sign_flip_p"] = exact_sign_flip_p(
        contribution_change_matrix
    )
    contribution_result["fdr"] = bh_adjust(
        contribution_result["sign_flip_p"].to_numpy()
    )
    return total_result, contribution_result


def select_figure_data(
    pathway_effects: pd.DataFrame,
    contributions: pd.DataFrame,
    paired_pathways: pd.DataFrame,
    infant_contributions: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    ranked_targets = (
        pathway_effects[pathway_effects["target_carbon_pathway"]]
        .sort_values(
            ["loo_sign_stability", "hedges_g"],
            ascending=[False, False],
            key=lambda series: series.abs()
            if series.name == "hedges_g"
            else series,
        )
        .head(10)
    )
    ranked_global = (
        pathway_effects.sort_values(
            ["loo_sign_stability", "hedges_g"],
            ascending=[False, False],
            key=lambda series: series.abs()
            if series.name == "hedges_g"
            else series,
        )
        .head(6)
    )
    selected = (
        pd.concat([ranked_targets, ranked_global], ignore_index=True)
        .drop_duplicates("pathway")
        .head(14)
        .copy()
    )
    selected["selection_reason"] = np.where(
        selected["target_carbon_pathway"],
        "prespecified carbon-metabolism pathway",
        "large infant-level pathway effect",
    )
    selected_paths = selected["pathway"].tolist()
    selected_contributions = contributions[
        contributions["pathway"].isin(selected_paths)
    ].copy()

    taxon_rank = (
        selected_contributions.groupby("taxon", observed=True)[
            "signed_attribution"
        ]
        .apply(lambda values: float(np.max(np.abs(values))))
        .sort_values(ascending=False)
    )
    named_taxa = [
        taxon
        for taxon in taxon_rank.index
        if taxon != "Unclassified contribution"
    ][:11]
    if "Unclassified contribution" in taxon_rank.index:
        named_taxa.append("Unclassified contribution")
    heatmap = selected_contributions[
        selected_contributions["taxon"].isin(named_taxa)
    ][
        [
            "pathway",
            "pathway_id",
            "pathway_name",
            "taxon",
            "signed_attribution",
            "hedges_g",
            "loo_sign_stability",
            "prevalent_infants",
        ]
    ].copy()
    complete_index = pd.MultiIndex.from_product(
        [selected_paths, named_taxa],
        names=["pathway", "taxon"],
    )
    heatmap = (
        heatmap.set_index(["pathway", "taxon"])
        .reindex(complete_index)
        .reset_index()
    )
    heatmap["signed_attribution"] = heatmap["signed_attribution"].fillna(0.0)
    heatmap["hedges_g"] = heatmap["hedges_g"].fillna(0.0)
    heatmap["loo_sign_stability"] = heatmap[
        "loo_sign_stability"
    ].fillna(0.0)
    pathway_label = selected.set_index("pathway")["pathway_name"]
    heatmap["pathway_name"] = heatmap["pathway"].map(pathway_label)

    phase_plane = selected.merge(
        paired_pathways[
            [
                "pathway",
                "mean_recovery_minus_acute_log1p_cpm",
                "paired_sign_concordance",
                "sign_flip_p",
            ]
        ],
        on="pathway",
        how="left",
    )
    dominant = (
        selected_contributions.assign(
            abs_attr=lambda frame: frame["signed_attribution"].abs()
        )
        .sort_values("abs_attr", ascending=False)
        .drop_duplicates("pathway")
        [["pathway", "taxon", "signed_attribution"]]
        .rename(
            columns={
                "taxon": "dominant_taxon",
                "signed_attribution": "dominant_signed_attribution",
            }
        )
    )
    phase_plane = phase_plane.merge(dominant, on="pathway", how="left")

    infant_matrix = infant_contributions[
        infant_contributions["pathway"].isin(selected_paths)
        & infant_contributions["taxon"].isin(named_taxa)
    ].copy()
    infant_matrix["log1p_contribution_cpm"] = np.log1p(
        infant_matrix["contribution_cpm"]
    )

    network = selected_contributions[
        selected_contributions["signed_attribution"].abs() >= 0.075
    ].copy()
    network = network[
        network["taxon"] != "Unclassified contribution"
    ].sort_values("signed_attribution", key=lambda values: values.abs(), ascending=False)
    network = network[
        [
            "taxon",
            "pathway",
            "pathway_name",
            "signed_attribution",
            "hedges_g",
            "loo_sign_stability",
            "prevalent_infants",
        ]
    ].head(40)
    return {
        "selected_pathways": selected,
        "heatmap": heatmap,
        "phase_plane": phase_plane,
        "infant_matrix": infant_matrix,
        "network": network,
    }


def main() -> None:
    manifest, local_files, inventory = synchronize_inputs()
    totals, contributions, audit = parse_humann(manifest, local_files)
    infant_totals, infant_contributions, infant_order, case_mask = (
        build_infant_matrices(totals, contributions)
    )
    if int(case_mask.sum()) != 7 or int((~case_mask).sum()) != 6:
        raise RuntimeError(
            f"Expected 7 NEC and 6 control infants, found "
            f"{case_mask.sum()} and {(~case_mask).sum()}"
        )
    pathway_effects = pathway_statistics(
        infant_totals,
        infant_order,
        case_mask,
    )
    contribution_effects = contribution_statistics(
        infant_contributions,
        infant_order,
        case_mask,
    )
    paired_pathways, paired_contributions = paired_statistics(
        totals,
        contributions,
    )
    figure_data = select_figure_data(
        pathway_effects,
        contribution_effects,
        paired_pathways,
        infant_contributions,
    )

    outputs: dict[str, pd.DataFrame] = {
        "scope_run_decomposition_audit.csv": audit,
        "scope_run_pathway_totals.csv": totals,
        "scope_run_species_pathway_contributions.csv": contributions,
        "scope_infant_pathway_totals.csv": infant_totals,
        "scope_infant_species_pathway_contributions.csv": infant_contributions,
        "scope_pathway_effects.csv": pathway_effects,
        "scope_species_pathway_effects.csv": contribution_effects,
        "scope_paired_pathway_changes.csv": paired_pathways,
        "scope_paired_species_pathway_changes.csv": paired_contributions,
    }
    for name, frame in outputs.items():
        frame.to_csv(ensure_local_write(RESULT_ROOT / name), index=False)

    source_outputs = {
        "ED_Figure_5a_species_pathway_attribution_heatmap.csv": figure_data[
            "heatmap"
        ],
        "ED_Figure_5b_pathway_phase_plane.csv": figure_data["phase_plane"],
        "ED_Figure_5c_infant_driver_matrix.csv": figure_data["infant_matrix"],
        "ED_Figure_5d_species_pathway_network.csv": figure_data["network"],
        "ED_Figure_5_selected_pathways.csv": figure_data[
            "selected_pathways"
        ],
    }
    for name, frame in source_outputs.items():
        frame.to_csv(
            ensure_local_write(SOURCE_DATA_ROOT / name),
            index=False,
        )

    named = contribution_effects[
        contribution_effects["taxon"] != "Unclassified contribution"
    ].copy()
    top_drivers = (
        named[named["target_carbon_pathway"]]
        .sort_values(
            "signed_attribution",
            key=lambda values: values.abs(),
            ascending=False,
        )
        .head(12)[
            [
                "pathway_name",
                "taxon",
                "signed_attribution",
                "delta_contribution_cpm",
                "hedges_g",
                "permutation_p",
                "fdr",
                "loo_sign_stability",
            ]
        ]
        .to_dict("records")
    )
    summary = {
        "status": "PASS",
        "analysis": "SCOPE-NEC",
        "analysis_version": "2026-07-26",
        "statistical_unit": "infant",
        "runs": int(totals["run"].nunique()),
        "infants": int(infant_totals["infant_id"].nunique()),
        "nec_infants": int(case_mask.sum()),
        "control_infants": int((~case_mask).sum()),
        "paired_nec_infants": 6,
        "pathways": int(pathway_effects.shape[0]),
        "taxa_including_unclassified": int(
            contribution_effects["taxon"].nunique()
        ),
        "species_pathway_features": int(contribution_effects.shape[0]),
        "scaled_pathway_run_records": int(audit["scaled_pathways"].sum()),
        "maximum_post_adjustment_closure_error_rpk": float(
            audit["maximum_closure_error_rpk"].max()
        ),
        "minimum_pathway_fdr": float(pathway_effects["fdr"].min()),
        "minimum_species_pathway_fdr": float(
            contribution_effects["fdr"].min()
        ),
        "selected_pathways_for_figure": int(
            figure_data["selected_pathways"].shape[0]
        ),
        "top_targeted_carbon_drivers": top_drivers,
        "input_files": int(inventory.shape[0]),
        "random_seed": SEED,
        "cloud_files_written": 0,
    }
    summary_path = ensure_local_write(RESULT_ROOT / "scope_analysis_summary.json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
