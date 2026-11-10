from __future__ import annotations

# Academic Figure Skill Typography Baseline -- COPY VERBATIM
import matplotlib as mpl
from matplotlib import font_manager
from pathlib import Path

font_manager.findfont("Liberation Sans", fallback_to_default=False)

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Liberation Sans"],
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 8,
    "figure.titlesize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "legend.frameon": False,
})

# Academic Figure Skill Nature/Cell/Science Color Palette -- COPY VERBATIM
CATEGORICAL = ["#2166AC", "#B2182B", "#1B7837", "#F1A340", "#762A83", "#666666"]
CATEGORICAL_EXTENDED = [
    "#2166AC", "#B2182B", "#1B7837", "#F1A340", "#762A83", "#666666",
    "#4393C3", "#D6604D", "#5AAE61", "#B35806", "#9970AB", "#999999",
]
DIVERGING = ["#2166AC", "#F7F7F7", "#B2182B"]
SEQUENTIAL = ["#F7FBFF", "#6BAED6", "#08306B"]
ACCENT_RED = "#B2182B"
GREY = "#999999"
BLACK = "#222222"

# Academic Figure Skill Export Baseline -- COPY VERBATIM
mpl.rcParams.update({
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
    "savefig.bbox": "tight",
    "savefig.dpi": 300,
})


def save_cns_figure(fig, filename):
    fig.savefig(f"{filename}.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(f"{filename}.png", bbox_inches="tight", dpi=300)


# Asset Confirmation
# - Manhattan/plot_Manhattan.py: inherited alternating chromosome rhythm and
#   rasterized point handling; significance threshold annotations were adapted.
# - heatmap/composition_heatmap.png and plot_composition.py: inherited square
#   cell geometry and external compact colorbars.
# - PCA/plot_PCA_1.png and plot_PCA.R: inherited equal-axis scatter geometry;
#   ellipses and grey crosslines were excluded.

import math
import textwrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


ANALYSIS_ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ANALYSIS_ROOT / "results" / "eclipse_nec"
SOURCE_ROOT = (
    ANALYSIS_ROOT
    / "source_data"
    / "Extended_Data_Figure_8_ECLIPSE_NEC"
)
FIGURE_ROOT = ANALYSIS_ROOT / "figures" / "extended_data"
LEGEND_ROOT = ANALYSIS_ROOT / "figure_legends"
FIGURE_STEM = FIGURE_ROOT / "Extended_Data_Figure_8_ECLIPSE_NEC"
FIGURE_WIDTH_MM = 183
FIGURE_HEIGHT_MM = 205
MM_TO_INCH = 1 / 25.4

mpl.rcParams["font.sans-serif"] = ["Liberation Sans"]
mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["axes.grid"] = False
mpl.rcParams["legend.fontsize"] = 5.3
mpl.rcParams["axes.titlesize"] = 7
mpl.rcParams["axes.labelsize"] = 7
mpl.rcParams["xtick.labelsize"] = 5.5
mpl.rcParams["ytick.labelsize"] = 5.5

OKABE_ITO = {
    "orange": "#E69F00",
    "sky_blue": "#56B4E9",
    "bluish_green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "reddish_purple": "#CC79A7",
    "black": "#000000",
    "white": "#FFFFFF",
}

DIVERGING_OKABE = LinearSegmentedColormap.from_list(
    "okabe_ito_diverging",
    [OKABE_ITO["blue"], OKABE_ITO["white"], OKABE_ITO["vermillion"]],
)

CONDITION_COLORS = {
    "control": OKABE_ITO["blue"],
    "nec": OKABE_ITO["vermillion"],
    "future_nec": OKABE_ITO["orange"],
}

CONDITION_LABELS = {
    "control": "Control",
    "nec": "NEC",
    "future_nec": "Future NEC",
}

MODULE_ORDER = [
    "Host-response circuit",
    "M-cell identity",
    "LEC identity",
    "ILC3 identity",
    "Resident T identity",
    "Treg identity",
    "Ferroptosis",
    "FXR-bile acid",
    "Hypoxia-epithelial injury",
]

STATE_ORDER = ["M cell", "LEC", "T/NK", "Treg"]
TISSUE_ORDER = ["ileum", "colon", "stool", "blood"]


def ensure_local(path: Path) -> Path:
    resolved = path.resolve()
    root = ANALYSIS_ROOT.resolve()
    if resolved != root and root not in resolved.parents:
        raise RuntimeError(f"Refusing write outside analysis root: {resolved}")
    return resolved


def panel_label(ax: mpl.axes.Axes, label: str) -> None:
    ax.text(
        -0.14,
        1.045,
        label,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        va="bottom",
        ha="right",
        color=OKABE_ITO["black"],
        clip_on=False,
    )


def wrap_label(value: str, width: int = 24) -> str:
    return textwrap.fill(
        str(value),
        width=width,
        break_long_words=False,
        break_on_hyphens=False,
    )


def clean_heatmap_axis(ax: mpl.axes.Axes) -> None:
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0, pad=1.5)


def add_colorbar(
    ax: mpl.axes.Axes,
    mesh,
    label: str,
    height: str = "55%",
) -> None:
    cax = inset_axes(
        ax,
        width="3.5%",
        height=height,
        loc="lower left",
        bbox_to_anchor=(1.035, 0.22, 1, 1),
        bbox_transform=ax.transAxes,
        borderpad=0,
    )
    colorbar = ax.figure.colorbar(mesh, cax=cax)
    colorbar.outline.set_linewidth(0.45)
    colorbar.ax.tick_params(labelsize=5, width=0.45, length=2)
    colorbar.set_label(label, fontsize=5.2, labelpad=2)


def chromosome_order(values: pd.Series) -> list[str]:
    def key(chromosome: str) -> tuple[int, str]:
        value = chromosome.replace("chr", "")
        if value.isdigit():
            return int(value), ""
        return 23 if value == "X" else 24, value

    return sorted(values.astype(str).unique(), key=key)


def manhattan_coordinates(meta: pd.DataFrame) -> pd.DataFrame:
    meta = meta.copy()
    chromosomes = chromosome_order(meta["chromosome"])
    offsets: dict[str, int] = {}
    cursor = 0
    midpoint_rows = []
    for chromosome in chromosomes:
        maximum = int(
            meta.loc[meta["chromosome"].eq(chromosome), "bin_start"].max()
        )
        offsets[chromosome] = cursor
        midpoint_rows.append(
            {
                "chromosome": chromosome,
                "midpoint": cursor + maximum / 2,
            }
        )
        cursor += maximum + 2_000_000
    meta["genome_position"] = (
        meta["bin_start"].to_numpy(dtype=np.int64)
        + meta["chromosome"].map(offsets).to_numpy(dtype=np.int64)
    )
    meta["minus_log10_p"] = -np.log10(
        np.clip(meta["meta_pvalue"].to_numpy(dtype=float), 1e-300, 1.0)
    )
    meta.attrs["midpoints"] = pd.DataFrame(midpoint_rows)
    return meta


def top_window_tissue_matrix(
    meta: pd.DataFrame,
    count: int = 12,
) -> pd.DataFrame:
    selected = meta.nsmallest(count, "meta_pvalue")[
        ["chromosome", "bin_start", "meta_pvalue", "meta_fdr"]
    ].copy()
    selected["window"] = selected.apply(
        lambda row: (
            f"{row['chromosome']}:{int(row['bin_start'] / 1_000_000)}."
            f"{int((row['bin_start'] % 1_000_000) / 1_000):03d} Mb"
        ),
        axis=1,
    )
    rows = []
    for tissue in TISSUE_ORDER:
        effects = pd.read_parquet(
            RESULT_ROOT / f"eclipse_{tissue}_window_effects.parquet",
            columns=["chromosome", "bin_start", "delta_beta"],
        )
        joined = selected.merge(
            effects,
            on=["chromosome", "bin_start"],
            how="left",
        )
        joined["tissue"] = tissue
        rows.append(joined)
    return pd.concat(rows, ignore_index=True)


def panel_a(ax: mpl.axes.Axes, meta: pd.DataFrame) -> None:
    midpoints = meta.attrs["midpoints"]
    chromosomes = chromosome_order(meta["chromosome"])
    for index, chromosome in enumerate(chromosomes):
        subset = meta[meta["chromosome"].eq(chromosome)]
        color = (
            OKABE_ITO["blue"]
            if index % 2 == 0
            else OKABE_ITO["sky_blue"]
        )
        ax.scatter(
            subset["genome_position"],
            subset["minus_log10_p"],
            s=0.45,
            c=color,
            alpha=0.55,
            linewidths=0,
            rasterized=True,
        )
    significant = meta[meta["meta_fdr"].lt(0.05)]
    ax.scatter(
        significant["genome_position"],
        significant["minus_log10_p"],
        s=17,
        c=OKABE_ITO["vermillion"],
        edgecolors=OKABE_ITO["black"],
        linewidths=0.35,
        zorder=5,
    )
    for row in significant.itertuples(index=False):
        ax.annotate(
            f"{row.chromosome}:{row.bin_start / 1_000_000:.3f} Mb",
            (row.genome_position, row.minus_log10_p),
            xytext=(3, 4),
            textcoords="offset points",
            fontsize=5,
            color=OKABE_ITO["black"],
        )
    ax.set_xticks(midpoints["midpoint"])
    ax.set_xticklabels(
        [value.replace("chr", "") for value in midpoints["chromosome"]],
        rotation=0,
    )
    ax.set_xlabel("Genomic position")
    ax.set_ylabel("−log₁₀ cross-tissue P")
    ax.set_title("Cross-tissue methylation meta-analysis", loc="left")
    ax.margins(x=0.005)
    ax.grid(False)
    panel_label(ax, "a")


def panel_b(ax: mpl.axes.Axes, tissue: pd.DataFrame) -> None:
    window_order = (
        tissue[["window", "meta_pvalue"]]
        .drop_duplicates()
        .sort_values("meta_pvalue")["window"]
        .tolist()
    )
    matrix = tissue.pivot(
        index="window",
        columns="tissue",
        values="delta_beta",
    ).reindex(index=window_order, columns=TISSUE_ORDER)
    limit = float(np.nanquantile(np.abs(matrix.to_numpy()), 0.97))
    limit = max(limit, 0.05)
    mesh = ax.imshow(
        matrix.to_numpy(),
        cmap=DIVERGING_OKABE,
        norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit),
        interpolation="nearest",
        aspect="equal",
    )
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels(
        [value.capitalize() for value in matrix.columns],
        rotation=45,
        ha="right",
    )
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index, fontsize=4.7)
    ax.set_title("Top-window tissue effects", loc="left")
    clean_heatmap_axis(ax)
    ax.set_box_aspect(len(matrix.index) / len(matrix.columns))
    add_colorbar(ax, mesh, "Δβ", height="46%")
    panel_label(ax, "b")


def panel_c(ax: mpl.axes.Axes, signals: pd.DataFrame) -> None:
    signals = signals.copy()
    signals["minus_log10_p"] = -np.log10(
        np.clip(signals["signal_acat_pvalue"], 1e-300, 1.0)
    )
    ax.scatter(
        signals["median_weighted_delta_beta"],
        signals["minus_log10_p"],
        s=np.clip(5 + 5 * signals["gene_count"], 5, 22),
        c=OKABE_ITO["sky_blue"],
        alpha=0.38,
        linewidths=0,
        rasterized=True,
    )
    top = signals.nsmallest(6, "signal_acat_pvalue")
    ax.scatter(
        top["median_weighted_delta_beta"],
        top["minus_log10_p"],
        s=np.clip(12 + 5 * top["gene_count"], 12, 28),
        c=OKABE_ITO["vermillion"],
        edgecolors=OKABE_ITO["black"],
        linewidths=0.4,
        zorder=4,
    )
    for row in top.itertuples(index=False):
        ax.annotate(
            row.representative_gene,
            (row.median_weighted_delta_beta, row.minus_log10_p),
            xytext=(3, 3),
            textcoords="offset points",
            fontsize=5,
        )
    ax.set_xlabel("Median cross-tissue Δβ")
    ax.set_ylabel("−log₁₀ promoter-signal P")
    ax.set_title("Overlapping promoter signals", loc="left")
    ax.grid(False)
    panel_label(ax, "c")


def panel_d(ax: mpl.axes.Axes, modules: pd.DataFrame) -> None:
    matrix = modules.pivot(
        index="target_state",
        columns="module",
        values="spearman_rho",
    ).reindex(index=STATE_ORDER, columns=MODULE_ORDER)
    mesh = ax.imshow(
        matrix.to_numpy(),
        cmap=DIVERGING_OKABE,
        norm=TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1),
        interpolation="nearest",
        aspect="equal",
    )
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels(
        [wrap_label(value, width=16) for value in matrix.columns],
        rotation=50,
        ha="right",
        rotation_mode="anchor",
    )
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index)
    ax.set_title("Promoter-expression coupling", loc="left")
    clean_heatmap_axis(ax)
    ax.set_box_aspect(len(matrix.index) / len(matrix.columns))
    add_colorbar(ax, mesh, "Spearman rho", height="62%")
    panel_label(ax, "d")


def panel_e(ax: mpl.axes.Axes, pca: pd.DataFrame) -> None:
    markers = {"control": "o", "nec": "^", "future_nec": "D"}
    for condition in ["control", "nec", "future_nec"]:
        subset = pca[pca["analysis_condition"].eq(condition)]
        ax.scatter(
            subset["PC1"],
            subset["PC2"],
            s=34,
            marker=markers[condition],
            facecolor=CONDITION_COLORS[condition],
            edgecolor=OKABE_ITO["black"],
            linewidth=0.45,
            alpha=0.9,
            label=CONDITION_LABELS[condition],
        )
    pc1 = float(pca["PC1_variance"].iloc[0]) * 100
    pc2 = float(pca["PC2_variance"].iloc[0]) * 100
    ax.set_xlabel(f"PC1 ({pc1:.1f}%)")
    ax.set_ylabel(f"PC2 ({pc2:.1f}%)")
    ax.set_title("Covariate-residualized stool methylome", loc="left")
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(0.0, -0.19),
        ncol=3,
        handletextpad=0.35,
        columnspacing=0.8,
        borderaxespad=0,
    )
    ax.grid(False)
    panel_label(ax, "e")


def panel_f(ax: mpl.axes.Axes, scores: pd.DataFrame) -> None:
    primary = scores[
        scores["analysis_condition"].isin(["control", "nec"])
    ].copy()
    rng = np.random.default_rng(20260726)
    positions = {"control": 0, "nec": 1}
    for condition in ["control", "nec"]:
        subset = primary[primary["analysis_condition"].eq(condition)]
        x = positions[condition] + rng.uniform(-0.08, 0.08, len(subset))
        values = subset["leave_one_subject_out_score"].to_numpy(dtype=float)
        ax.scatter(
            x,
            values,
            s=38,
            marker="o" if condition == "control" else "^",
            facecolor=CONDITION_COLORS[condition],
            edgecolor=OKABE_ITO["black"],
            linewidth=0.45,
            zorder=3,
        )
        median = float(np.median(values))
        q1, q3 = np.quantile(values, [0.25, 0.75])
        ax.vlines(
            positions[condition],
            q1,
            q3,
            color=OKABE_ITO["black"],
            linewidth=1.6,
            zorder=2,
        )
        ax.hlines(
            median,
            positions[condition] - 0.13,
            positions[condition] + 0.13,
            color=OKABE_ITO["black"],
            linewidth=1.0,
            zorder=4,
        )
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Control", "NEC"])
    ax.set_xlim(-0.45, 1.45)
    ax.set_ylabel("Nested leave-one-subject-out score")
    ax.set_title("Age-adjusted NEC-axis discrimination", loc="left")
    ax.text(
        0.98,
        0.96,
        "AUC 0.70",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=6,
        color=OKABE_ITO["black"],
    )
    ax.grid(False)
    panel_label(ax, "f")


def save_all_formats(fig: mpl.figure.Figure, stem: Path) -> None:
    ensure_local(stem)
    save_cns_figure(fig, str(stem))
    fig.savefig(f"{stem}.svg", bbox_inches="tight", dpi=300)
    fig.savefig(
        f"{stem}.tiff",
        bbox_inches="tight",
        dpi=600,
        pil_kwargs={"compression": "tiff_lzw"},
    )


def write_legend() -> None:
    legend = """# Extended Data Figure 8 | ECLIPSE-NEC links cross-tissue methylation to intestinal cell states

**a,** Genome-wide cross-tissue methylation meta-analysis across ileum, colon, stool and blood. The analysis includes 80 public Bismark coverage samples and 890,377 1-kb windows represented in at least two tissues. Vermillion points mark the two windows with Benjamini-Hochberg FDR below 0.05.

**b,** Tissue-resolved methylation effects for the 12 windows with the smallest cross-tissue P values. Positive values indicate greater methylation in NEC. Stool effects derive from an M-value model adjusted for corrected gestational age, postnatal day, sex and race.

**c,** Cross-tissue promoter signals after genes with overlapping plus or minus 2-kb promoters were collapsed into 41,280 genomic signal units. Point area denotes the number of overlapping gene annotations; labels identify the six smallest signal-level P values. The minimum signal-level FDR was 0.058.

**d,** Correlation between promoter activation effects, defined as the negative cross-tissue methylation effect, and strict single-cell pseudobulk expression changes across prespecified cell-state programs. Gene sets containing at most nine mapped genes used all gene-label permutations; larger sets used the asymptotic Spearman test. No module correlation remained significant after FDR correction.

**e,** Principal-component representation of 500 covariate-residualized stool methylation windows. Colors and shapes identify control, NEC and samples collected before NEC onset. PC1 and PC2 explain 45.8% and 21.0% of residualized variation, respectively.

**f,** Nested leave-one-subject-out NEC-axis scores for the 13 primary stool samples. Feature selection, covariate residualization and axis estimation were repeated without the held-out subject. Vertical lines span the interquartile range and horizontal lines denote medians; the cross-validated area under the receiver-operating-characteristic curve was 0.70.

Ileum, colon and blood window-level inference used sample-level Welch tests; stool used feature-wise covariate-adjusted OLS. Cross-tissue window effects used weighted Stouffer integration across independent subjects. Promoter gene annotations sharing overlapping intervals were analyzed as one signal for multiple-testing correction. Liberation Sans and the Okabe-Ito colorblind-safe palette are used throughout.
"""
    ensure_local(
        LEGEND_ROOT / "Extended_Data_Figure_8_ECLIPSE_NEC_legend.md"
    ).write_text(legend, encoding="utf-8")


def main() -> None:
    for directory in [FIGURE_ROOT, LEGEND_ROOT, SOURCE_ROOT]:
        ensure_local(directory).mkdir(parents=True, exist_ok=True)

    meta_raw = pd.read_parquet(
        RESULT_ROOT / "eclipse_cross_tissue_window_meta.parquet"
    )
    meta = manhattan_coordinates(meta_raw)
    tissue = top_window_tissue_matrix(meta_raw, count=12)
    signals = pd.read_csv(
        RESULT_ROOT / "eclipse_cross_tissue_promoter_signal_meta.csv"
    )
    modules = pd.read_csv(
        RESULT_ROOT / "eclipse_cell_state_module_summary.csv"
    )
    pca = pd.read_csv(
        RESULT_ROOT / "eclipse_stool_methylation_pca.csv"
    )
    scores = pd.read_csv(
        RESULT_ROOT / "eclipse_stool_nec_axis_projection.csv"
    )

    meta_source = meta[
        [
            "chromosome",
            "bin_start",
            "bin_end",
            "meta_pvalue",
            "meta_fdr",
            "weighted_delta_beta",
            "tissues",
            "sign_concordance",
            "genome_position",
            "minus_log10_p",
        ]
    ].copy()
    meta_source.attrs = {}
    meta_source.to_parquet(
        ensure_local(
            SOURCE_ROOT / "ED_Figure_8a_cross_tissue_windows.parquet"
        ),
        index=False,
    )
    tissue.to_csv(
        ensure_local(
            SOURCE_ROOT / "ED_Figure_8b_top_window_tissue_effects.csv"
        ),
        index=False,
    )
    signals.to_csv(
        ensure_local(
            SOURCE_ROOT / "ED_Figure_8c_promoter_signals.csv"
        ),
        index=False,
    )
    modules.to_csv(
        ensure_local(
            SOURCE_ROOT / "ED_Figure_8d_module_concordance.csv"
        ),
        index=False,
    )
    pca.to_csv(
        ensure_local(
            SOURCE_ROOT / "ED_Figure_8e_stool_residual_pca.csv"
        ),
        index=False,
    )
    scores.to_csv(
        ensure_local(
            SOURCE_ROOT / "ED_Figure_8f_stool_nested_scores.csv"
        ),
        index=False,
    )

    fig = plt.figure(
        figsize=(
            FIGURE_WIDTH_MM * MM_TO_INCH,
            FIGURE_HEIGHT_MM * MM_TO_INCH,
        ),
        constrained_layout=True,
    )
    grid = fig.add_gridspec(
        3,
        2,
        width_ratios=[1.38, 1.0],
        height_ratios=[1.0, 0.95, 0.9],
        hspace=0.20,
        wspace=0.22,
    )
    axes = [
        fig.add_subplot(grid[0, 0]),
        fig.add_subplot(grid[0, 1]),
        fig.add_subplot(grid[1, 0]),
        fig.add_subplot(grid[1, 1]),
        fig.add_subplot(grid[2, 0]),
        fig.add_subplot(grid[2, 1]),
    ]
    panel_a(axes[0], meta)
    panel_b(axes[1], tissue)
    panel_c(axes[2], signals)
    panel_d(axes[3], modules)
    panel_e(axes[4], pca)
    panel_f(axes[5], scores)
    save_all_formats(fig, FIGURE_STEM)
    plt.close(fig)
    write_legend()
    print(
        "ECLIPSE_FIGURE_COMPLETE "
        f"windows={len(meta)} signals={len(signals)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
