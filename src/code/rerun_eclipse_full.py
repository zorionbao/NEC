"""Run all ECLIPSE models with a verified vectorized Welch implementation.

This wrapper changes no data filters, hypotheses, random seeds or sample units.
It recomputes every tissue cohort and then executes stages 24, 25 and 26.
"""
from pathlib import Path
from types import SimpleNamespace
import argparse, importlib, json, subprocess, sys, warnings
import numpy as np
from scipy import stats


def welch(a, b, axis=1, equal_var=False, nan_policy="omit", **kwargs):
    if axis != 1 or equal_var or nan_policy != "omit" or kwargs:
        raise ValueError("Vectorized implementation supports the specified Welch model only")
    a, b = np.asarray(a, float), np.asarray(b, float)
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        n1, n0 = np.isfinite(a).sum(1), np.isfinite(b).sum(1)
        v1, v0 = np.nanvar(a, axis=1, ddof=1), np.nanvar(b, axis=1, ddof=1)
        u1, u0 = v1 / n1, v0 / n0
        den = u1 + u0
        t = (np.nanmean(a, 1) - np.nanmean(b, 1)) / np.sqrt(den)
        df = den ** 2 / (u1 ** 2 / (n1 - 1) + u0 ** 2 / (n0 - 1))
        df = np.where(np.isnan(df), 1, df)
        p = 2 * stats.t.sf(np.abs(t), df)
    return SimpleNamespace(statistic=t, pvalue=p, df=df)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    a = p.parse_args()
    root = a.root.resolve()
    original = stats.ttest_ind
    rng = np.random.default_rng(20261106)
    x, y = rng.normal(size=(1000, 9)), rng.normal(size=(1000, 7))
    x[rng.random(x.shape) < .2] = np.nan
    y[rng.random(y.shape) < .2] = np.nan
    expected = original(x, y, axis=1, equal_var=False, nan_policy="omit").pvalue
    observed = welch(x, y).pvalue
    delta = float(np.nanmax(np.abs(expected - observed)))
    if not np.allclose(expected, observed, atol=1e-12, rtol=1e-12, equal_nan=True):
        raise RuntimeError("Vectorized Welch equivalence check failed")
    audit_path = root / "logs" / "round2_welch_equivalence.json"
    audit = {
        "rows": 1000, "max_absolute_p_difference": delta,
        "all_cohorts_recomputed": False, "uses_cached_cohort_effects": False,
        "models_filters_and_seeds_unchanged": True,
        "status": "numerical_equivalence_verified; computation_pending"}
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2))
    sys.path.insert(0, str(root / "legacy_analysis" / "code"))
    module = importlib.import_module("16_eclipse_nec")
    module.stats.ttest_ind = welch
    module.main()
    for name in ["24_recompute_eclipse_empirical.py", "25_eclipse_stool_covariate_adjustment.py", "26_eclipse_promoter_signal_collapse.py"]:
        subprocess.run([sys.executable, str(root / "legacy_analysis" / "code" / name)], cwd=root, check=True)
    audit.update(all_cohorts_recomputed=True, status="completed")
    audit_path.write_text(json.dumps(audit, indent=2))
    (root / "logs" / "methylation_complete.done").touch()


if __name__ == "__main__":
    main()
