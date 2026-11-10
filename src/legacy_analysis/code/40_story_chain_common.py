from __future__ import annotations

import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy import stats


SEED = 20260727
CODE_ROOT = Path(__file__).resolve().parent
ANALYSIS_ROOT = CODE_ROOT.parent
PACKAGE_ROOT = ANALYSIS_ROOT.parent
WORKSPACE_ROOT = PACKAGE_ROOT
LOCAL_REVISION_ROOT = ANALYSIS_ROOT.parent
PACKAGE_RESULTS_ROOT = ANALYSIS_ROOT / "results" / "story_chain_upgrade"
LOCAL_RESULTS_ROOT = LOCAL_REVISION_ROOT / "12_story_chain_upgrade"


def output_roots(module: str) -> tuple[Path, Path]:
    package = PACKAGE_RESULTS_ROOT / module
    local = LOCAL_RESULTS_ROOT / module
    package.mkdir(parents=True, exist_ok=True)
    local.mkdir(parents=True, exist_ok=True)
    return package, local


def write_csv(frame: pd.DataFrame, module: str, filename: str) -> tuple[Path, Path]:
    outputs = []
    for root in output_roots(module):
        path = root / filename
        frame.to_csv(path, index=False, encoding="utf-8")
        outputs.append(path)
    return outputs[0], outputs[1]


def write_json(payload: dict[str, Any], module: str, filename: str) -> tuple[Path, Path]:
    outputs = []
    for root in output_roots(module):
        path = root / filename
        path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        outputs.append(path)
    return outputs[0], outputs[1]


def sha256(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def bh_fdr(values: Iterable[float]) -> np.ndarray:
    p = np.asarray(list(values), dtype=float)
    out = np.full(p.shape, np.nan, dtype=float)
    finite = np.isfinite(p)
    if not finite.any():
        return out
    ranked = p[finite]
    order = np.argsort(ranked)
    adjusted = np.empty_like(ranked)
    adjusted[order] = np.minimum.accumulate(
        (ranked[order] * len(ranked) / np.arange(1, len(ranked) + 1))[::-1]
    )[::-1]
    out[finite] = np.clip(adjusted, 0.0, 1.0)
    return out


def hedges_g(case: np.ndarray, control: np.ndarray) -> float:
    case = np.asarray(case, dtype=float)
    control = np.asarray(control, dtype=float)
    case = case[np.isfinite(case)]
    control = control[np.isfinite(control)]
    if len(case) < 2 or len(control) < 2:
        return np.nan
    degrees = len(case) + len(control) - 2
    pooled_variance = (
        (len(case) - 1) * np.var(case, ddof=1)
        + (len(control) - 1) * np.var(control, ddof=1)
    ) / degrees
    if pooled_variance <= 0:
        return 0.0
    correction = 1.0 - 3.0 / (4.0 * degrees - 1.0)
    return correction * (np.mean(case) - np.mean(control)) / math.sqrt(pooled_variance)


def exact_label_permutation(
    values: np.ndarray,
    labels: np.ndarray,
    statistic: str = "hedges_g",
) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    labels = np.asarray(labels, dtype=bool)
    finite = np.isfinite(values)
    values = values[finite]
    labels = labels[finite]
    n_case = int(labels.sum())
    if n_case < 2 or len(values) - n_case < 2:
        return {
            "effect": np.nan,
            "p_value": np.nan,
            "assignments": 0,
        }

    def evaluate(mask: np.ndarray) -> float:
        if statistic == "mean_difference":
            return float(values[mask].mean() - values[~mask].mean())
        return float(hedges_g(values[mask], values[~mask]))

    observed = evaluate(labels)
    null = []
    for selected in itertools.combinations(range(len(values)), n_case):
        mask = np.zeros(len(values), dtype=bool)
        mask[list(selected)] = True
        null.append(evaluate(mask))
    null_values = np.asarray(null, dtype=float)
    p_value = float(
        np.mean(np.abs(null_values) >= abs(observed) - np.finfo(float).eps)
    )
    return {
        "effect": observed,
        "p_value": p_value,
        "assignments": int(len(null_values)),
    }


def residualize(values: np.ndarray, covariates: np.ndarray) -> np.ndarray:
    y = np.asarray(values, dtype=float)
    x = np.asarray(covariates, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    design = np.column_stack([np.ones(len(y)), x])
    return y - design @ (np.linalg.pinv(design) @ y)


def leave_one_out_effects(
    frame: pd.DataFrame,
    outcome: str,
    covariates: list[str] | None = None,
) -> pd.DataFrame:
    covariates = covariates or []
    rows: list[dict[str, Any]] = []
    for donor_id in frame["donor_id"].tolist():
        work = frame.loc[frame["donor_id"].ne(donor_id)].dropna(
            subset=[outcome, "condition", *covariates]
        )
        values = work[outcome].to_numpy(dtype=float)
        if covariates:
            values = residualize(values, work[covariates].to_numpy(dtype=float))
        result = exact_label_permutation(
            values,
            work["condition"].eq("nec").to_numpy(),
        )
        rows.append(
            {
                "omitted_donor_id": donor_id,
                "effect": result["effect"],
                "p_value": result["p_value"],
                "assignments": result["assignments"],
            }
        )
    return pd.DataFrame(rows)


def bootstrap_ci(
    case: np.ndarray,
    control: np.ndarray,
    iterations: int = 5000,
    seed: int = SEED,
) -> tuple[float, float]:
    case = np.asarray(case, dtype=float)
    control = np.asarray(control, dtype=float)
    rng = np.random.default_rng(seed)
    draws = np.empty(iterations, dtype=float)
    for index in range(iterations):
        case_draw = rng.choice(case, size=len(case), replace=True)
        control_draw = rng.choice(control, size=len(control), replace=True)
        draws[index] = hedges_g(case_draw, control_draw)
    return tuple(np.nanquantile(draws, [0.025, 0.975]).tolist())


def p_to_two_sided_z(p_value: float, sign: float = 1.0) -> float:
    if not np.isfinite(p_value):
        return np.nan
    p_value = float(np.clip(p_value, np.finfo(float).tiny, 1.0))
    return float(np.sign(sign) * stats.norm.isf(p_value / 2.0))


def effect_se_from_p(effect: float, p_value: float) -> float:
    z = abs(p_to_two_sided_z(p_value, effect))
    if not np.isfinite(effect) or not np.isfinite(z) or z <= 0:
        return np.nan
    return abs(float(effect)) / z


def random_effects_meta(effect: np.ndarray, standard_error: np.ndarray) -> dict[str, float]:
    y = np.asarray(effect, dtype=float)
    se = np.asarray(standard_error, dtype=float)
    finite = np.isfinite(y) & np.isfinite(se) & (se > 0)
    y = y[finite]
    se = se[finite]
    k = len(y)
    if k < 2:
        return {
            "k": k,
            "effect": np.nan,
            "standard_error": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "p_value": np.nan,
            "tau2": np.nan,
            "i2": np.nan,
            "sign_probability": np.nan,
        }
    variance = se**2
    fixed_weights = 1.0 / variance
    fixed_mean = float(np.sum(fixed_weights * y) / np.sum(fixed_weights))
    q = float(np.sum(fixed_weights * (y - fixed_mean) ** 2))
    c_value = float(
        np.sum(fixed_weights)
        - np.sum(fixed_weights**2) / np.sum(fixed_weights)
    )
    tau2 = max(0.0, (q - (k - 1)) / c_value) if c_value > 0 else 0.0
    weights = 1.0 / (variance + tau2)
    pooled = float(np.sum(weights * y) / np.sum(weights))
    pooled_se = float(math.sqrt(1.0 / np.sum(weights)))
    z_value = pooled / pooled_se
    p_value = float(2.0 * stats.norm.sf(abs(z_value)))
    i2 = max(0.0, (q - (k - 1)) / q) * 100.0 if q > 0 else 0.0
    sign_probability = float(
        stats.norm.cdf(abs(pooled) / pooled_se)
    )
    return {
        "k": k,
        "effect": pooled,
        "standard_error": pooled_se,
        "ci_low": pooled - 1.96 * pooled_se,
        "ci_high": pooled + 1.96 * pooled_se,
        "p_value": p_value,
        "tau2": tau2,
        "i2": i2,
        "sign_probability": sign_probability,
    }
