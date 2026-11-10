from __future__ import annotations

import itertools
import math
from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd


RANDOM_SEED = 20260724


def bh_fdr(p_values: Iterable[float]) -> np.ndarray:
    """Benjamini-Hochberg adjustment with the required reverse monotonic step."""
    values = np.asarray(list(p_values), dtype=float)
    adjusted = np.full(values.shape, np.nan, dtype=float)
    valid = np.isfinite(values)
    if not valid.any():
        return adjusted
    p = values[valid]
    order = np.argsort(p, kind="mergesort")
    ranked = p[order]
    scaled = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    monotone = np.minimum.accumulate(scaled[::-1])[::-1]
    restored = np.empty_like(monotone)
    restored[order] = np.clip(monotone, 0.0, 1.0)
    adjusted[valid] = restored
    return adjusted


def hedges_g(case: Iterable[float], control: Iterable[float]) -> float:
    case_values = np.asarray(list(case), dtype=float)
    control_values = np.asarray(list(control), dtype=float)
    case_values = case_values[np.isfinite(case_values)]
    control_values = control_values[np.isfinite(control_values)]
    n_case = len(case_values)
    n_control = len(control_values)
    if n_case < 2 or n_control < 2:
        return float("nan")
    variance = (
        (n_case - 1) * np.var(case_values, ddof=1)
        + (n_control - 1) * np.var(control_values, ddof=1)
    ) / (n_case + n_control - 2)
    if not np.isfinite(variance) or variance <= 0:
        return 0.0 if np.isclose(case_values.mean(), control_values.mean()) else float("nan")
    cohen_d = (case_values.mean() - control_values.mean()) / math.sqrt(variance)
    degrees = n_case + n_control - 2
    correction = 1.0 - 3.0 / (4.0 * degrees - 1.0) if degrees > 1 else 1.0
    return float(cohen_d * correction)


def exact_group_permutation(
    values: Iterable[float],
    labels: Iterable[Any],
    case_label: Any = "nec",
    max_exact: int = 200_000,
    random_seed: int = RANDOM_SEED,
) -> dict[str, float | int | str]:
    values_array = np.asarray(list(values), dtype=float)
    labels_array = np.asarray(list(labels), dtype=object)
    keep = np.isfinite(values_array) & pd.notna(labels_array)
    values_array = values_array[keep]
    labels_array = labels_array[keep]
    case_mask = labels_array == case_label
    n_case = int(case_mask.sum())
    n_total = len(values_array)
    n_control = n_total - n_case
    if n_case < 1 or n_control < 1:
        return {
            "observed_delta": float("nan"),
            "p_value": float("nan"),
            "n_permutations": 0,
            "method": "insufficient_groups",
        }
    observed = float(values_array[case_mask].mean() - values_array[~case_mask].mean())
    total_combinations = math.comb(n_total, n_case)
    rng = np.random.default_rng(random_seed)
    if total_combinations <= max_exact:
        assignments = itertools.combinations(range(n_total), n_case)
        exceed = 0
        evaluated = 0
        overall_sum = values_array.sum()
        for assignment in assignments:
            case_sum = float(values_array[list(assignment)].sum())
            difference = case_sum / n_case - (overall_sum - case_sum) / n_control
            exceed += int(abs(difference) >= abs(observed) - 1e-12)
            evaluated += 1
        p_value = exceed / evaluated
        method = "exact_label_permutation"
    else:
        evaluated = max_exact
        exceed = 0
        for _ in range(evaluated):
            assignment = rng.choice(n_total, size=n_case, replace=False)
            sampled_case = np.zeros(n_total, dtype=bool)
            sampled_case[assignment] = True
            difference = float(
                values_array[sampled_case].mean()
                - values_array[~sampled_case].mean()
            )
            exceed += int(abs(difference) >= abs(observed) - 1e-12)
        p_value = (exceed + 1) / (evaluated + 1)
        method = "monte_carlo_label_permutation"
    return {
        "observed_delta": observed,
        "p_value": float(p_value),
        "n_permutations": int(evaluated),
        "method": method,
    }


def exact_matrix_permutation(
    matrix: np.ndarray,
    labels: Iterable[Any],
    case_label: Any = "nec",
    max_exact: int = 200_000,
    random_seed: int = RANDOM_SEED,
) -> tuple[np.ndarray, np.ndarray, int, str]:
    values = np.asarray(matrix, dtype=float)
    labels_array = np.asarray(list(labels), dtype=object)
    if values.ndim != 2 or values.shape[0] != len(labels_array):
        raise ValueError("matrix must be observations by features and match labels")
    finite_rows = np.isfinite(values).all(axis=1) & pd.notna(labels_array)
    values = values[finite_rows]
    labels_array = labels_array[finite_rows]
    observed_mask = labels_array == case_label
    n_case = int(observed_mask.sum())
    n_total = len(labels_array)
    n_control = n_total - n_case
    if n_case < 1 or n_control < 1:
        return (
            np.full(values.shape[1], np.nan),
            np.full(values.shape[1], np.nan),
            0,
            "insufficient_groups",
        )
    observed = (
        values[observed_mask].mean(axis=0)
        - values[~observed_mask].mean(axis=0)
    )
    total_combinations = math.comb(n_total, n_case)
    if total_combinations <= max_exact:
        combinations = list(itertools.combinations(range(n_total), n_case))
        assignment = np.zeros((len(combinations), n_total), dtype=np.float32)
        for row_index, indices in enumerate(combinations):
            assignment[row_index, list(indices)] = 1.0
        method = "exact_label_permutation"
    else:
        rng = np.random.default_rng(random_seed)
        assignment = np.zeros((max_exact, n_total), dtype=np.float32)
        for row_index in range(max_exact):
            indices = rng.choice(n_total, size=n_case, replace=False)
            assignment[row_index, indices] = 1.0
        method = "monte_carlo_label_permutation"
    case_sums = assignment @ values
    total_sums = values.sum(axis=0, keepdims=True)
    permutations = case_sums / n_case - (total_sums - case_sums) / n_control
    exceed = (np.abs(permutations) >= np.abs(observed)[None, :] - 1e-12).sum(axis=0)
    if method == "exact_label_permutation":
        p_values = exceed / assignment.shape[0]
    else:
        p_values = (exceed + 1) / (assignment.shape[0] + 1)
    return observed, p_values.astype(float), assignment.shape[0], method


def bootstrap_hedges_g(
    case: Iterable[float],
    control: Iterable[float],
    n_bootstrap: int = 4000,
    random_seed: int = RANDOM_SEED,
) -> tuple[float, float]:
    case_values = np.asarray(list(case), dtype=float)
    control_values = np.asarray(list(control), dtype=float)
    case_values = case_values[np.isfinite(case_values)]
    control_values = control_values[np.isfinite(control_values)]
    if len(case_values) < 2 or len(control_values) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(random_seed)
    boot = np.empty(n_bootstrap, dtype=float)
    for index in range(n_bootstrap):
        boot[index] = hedges_g(
            rng.choice(case_values, size=len(case_values), replace=True),
            rng.choice(control_values, size=len(control_values), replace=True),
        )
    boot = boot[np.isfinite(boot)]
    if len(boot) == 0:
        return float("nan"), float("nan")
    return tuple(np.quantile(boot, [0.025, 0.975]).astype(float))


def paired_sign_flip(
    deltas: Iterable[float],
) -> dict[str, float | int | str]:
    values = np.asarray(list(deltas), dtype=float)
    values = values[np.isfinite(values)]
    n = len(values)
    if n == 0:
        return {
            "mean_delta": float("nan"),
            "p_value": float("nan"),
            "n_pairs": 0,
            "method": "insufficient_pairs",
        }
    observed = float(values.mean())
    signs = np.asarray(list(itertools.product((-1.0, 1.0), repeat=n)), dtype=float)
    permuted = (signs * values[None, :]).mean(axis=1)
    p_value = float((np.abs(permuted) >= abs(observed) - 1e-12).mean())
    return {
        "mean_delta": observed,
        "p_value": p_value,
        "n_pairs": n,
        "method": "exact_sign_flip",
    }


def effect_row(
    values: Iterable[float],
    labels: Iterable[Any],
    feature: str,
    family: str,
    case_label: Any = "nec",
    n_bootstrap: int = 4000,
    random_seed: int = RANDOM_SEED,
) -> dict[str, Any]:
    value_array = np.asarray(list(values), dtype=float)
    label_array = np.asarray(list(labels), dtype=object)
    keep = np.isfinite(value_array) & pd.notna(label_array)
    value_array = value_array[keep]
    label_array = label_array[keep]
    case = value_array[label_array == case_label]
    control = value_array[label_array != case_label]
    permutation = exact_group_permutation(
        value_array,
        label_array,
        case_label=case_label,
        random_seed=random_seed,
    )
    confidence_low, confidence_high = bootstrap_hedges_g(
        case,
        control,
        n_bootstrap=n_bootstrap,
        random_seed=random_seed,
    )
    return {
        "family": family,
        "feature": feature,
        "case_label": str(case_label),
        "n_case": int(len(case)),
        "n_control": int(len(control)),
        "case_mean": float(np.mean(case)) if len(case) else float("nan"),
        "control_mean": float(np.mean(control)) if len(control) else float("nan"),
        "mean_delta": permutation["observed_delta"],
        "hedges_g": hedges_g(case, control),
        "hedges_g_ci_low": confidence_low,
        "hedges_g_ci_high": confidence_high,
        "p_value": permutation["p_value"],
        "permutation_method": permutation["method"],
        "n_permutations": permutation["n_permutations"],
    }


def safe_zscore(values: Iterable[float]) -> np.ndarray:
    array = np.asarray(list(values), dtype=float)
    mean = np.nanmean(array)
    standard_deviation = np.nanstd(array, ddof=1)
    if not np.isfinite(standard_deviation) or standard_deviation == 0:
        return np.zeros_like(array, dtype=float)
    return (array - mean) / standard_deviation

