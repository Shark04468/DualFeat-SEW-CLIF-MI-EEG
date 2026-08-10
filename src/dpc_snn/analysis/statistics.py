"""Subject-level statistics and multiple-comparison helpers."""

from __future__ import annotations

import math
import contextlib
import io
import warnings
from typing import Any

import numpy as np


def paired_differences(
    rows: list[dict[str, Any]],
    model_a: str,
    model_b: str,
    metric: str,
    unit: str | tuple[str, ...] = ("dataset", "protocol", "subject", "seed"),
) -> np.ndarray:
    by_unit: dict[Any, dict[str, float]] = {}
    for row in rows:
        fields = (unit,) if isinstance(unit, str) else unit
        key = tuple(str(row.get(field, "")) for field in fields)
        by_unit.setdefault(key, {})
        by_unit[key][row["model"]] = float(row[metric])
    diffs = []
    for values in by_unit.values():
        if model_a in values and model_b in values:
            diffs.append(values[model_a] - values[model_b])
    return np.asarray(diffs, dtype=float)


def wilcoxon_signed_rank(diffs: np.ndarray) -> dict[str, float]:
    diffs = np.asarray(diffs, dtype=float)
    diffs = diffs[np.isfinite(diffs)]
    diffs = diffs[~np.isclose(diffs, 0.0)]
    if diffs.size == 0:
        return {"statistic": math.nan, "p_value": math.nan, "n": 0}
    try:
        from scipy.stats import wilcoxon

        stat, p = wilcoxon(diffs)
        return {"statistic": float(stat), "p_value": float(p), "n": int(diffs.size)}
    except ImportError:
        # Fallback sign-test approximation.
        positives = int((diffs > 0).sum())
        n = int(diffs.size)
        k = min(positives, n - positives)
        p = 2.0 * sum(math.comb(n, i) for i in range(k + 1)) * (0.5**n)
        return {"statistic": float(positives), "p_value": float(min(1.0, p)), "n": n}


def cohens_d_paired(diffs: np.ndarray) -> float:
    diffs = np.asarray(diffs, dtype=float)
    diffs = diffs[np.isfinite(diffs)]
    if diffs.size < 2:
        return math.nan
    std = diffs.std(ddof=1)
    return float(diffs.mean() / std) if std > 0 else math.nan


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size == 0 or y.size == 0:
        return math.nan
    gt = sum(float(a > b) for a in x for b in y)
    lt = sum(float(a < b) for a in x for b in y)
    return float((gt - lt) / (x.size * y.size))


def fdr_bh(p_values: list[float]) -> list[float]:
    p = np.asarray(p_values, dtype=float)
    valid = np.isfinite(p)
    if not valid.any():
        return [math.nan for _ in p_values]
    adjusted_full = np.full_like(p, np.nan, dtype=float)
    p_valid = p[valid]
    n = p_valid.size
    order = np.argsort(p_valid)
    ranked = p_valid[order]
    adjusted = np.empty(n, dtype=float)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        val = min(prev, ranked[i] * n / rank)
        adjusted[i] = val
        prev = val
    out = np.empty(n, dtype=float)
    out[order] = adjusted
    adjusted_full[valid] = out
    return adjusted_full.tolist()


def subject_level_summary(rows: list[dict[str, Any]], metric: str) -> dict[str, float]:
    vals = np.asarray([float(r[metric]) for r in rows if metric in r], dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return {"mean": math.nan, "std": math.nan, "n": 0}
    return {"mean": float(vals.mean()), "std": float(vals.std(ddof=1)) if vals.size > 1 else 0.0, "n": int(vals.size)}


def mixed_effects_or_fallback(rows: list[dict[str, Any]], metric: str = "accuracy") -> list[dict[str, Any]]:
    """Run a mixed-effects model when statsmodels is available.

    Fallback returns model-level subject-aggregated means and standard errors,
    which is still auditable and avoids fabricating a mixed-effect result.
    """

    try:
        with contextlib.redirect_stderr(io.StringIO()):
            import pandas as pd
    except Exception:
        grouped: dict[str, list[float]] = {}
        for row in rows:
            if metric in row and "model" in row:
                grouped.setdefault(str(row["model"]), []).append(float(row[metric]))
        return [
            {
                "analysis": "fallback_group_summary",
                "model": model,
                "metric": metric,
                "mean": float(np.nanmean(values)),
                "std": float(np.nanstd(values, ddof=1)) if len(values) > 1 else 0.0,
                "n": len(values),
            }
            for model, values in grouped.items()
        ]

    df = pd.DataFrame(rows)
    required = {"model", metric}
    if not required.issubset(df.columns):
        return [{"analysis": "mixed_effects", "status": "missing_required_columns", "required": "model," + metric}]
    if "subject" not in df.columns:
        df["subject"] = "unknown"
    if "dataset" not in df.columns:
        df["dataset"] = "unknown"
    df[metric] = pd.to_numeric(df[metric], errors="coerce")
    df["subject"] = df["subject"].astype(str)
    df["dataset"] = df["dataset"].astype(str)
    df["model"] = df["model"].astype(str)
    df = df.dropna(subset=[metric, "model", "subject"])
    if df.empty:
        return [{"analysis": "mixed_effects", "status": "no_valid_rows"}]
    try:
        import statsmodels.formula.api as smf

        formula = f"{metric} ~ C(model)"
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            fit = smf.mixedlm(formula, df, groups=df["subject"]).fit(
                reml=False, method="lbfgs", maxiter=200
            )
        if not bool(getattr(fit, "converged", False)):
            raise RuntimeError("Mixed-effects optimizer did not converge")
        inferential = np.concatenate(
            (
                np.asarray(fit.params, dtype=float),
                np.asarray(fit.bse, dtype=float),
                np.asarray(fit.pvalues, dtype=float),
            )
        )
        if not np.isfinite(inferential).all():
            raise FloatingPointError("Mixed-effects fit returned non-finite inference")
        return [
            {
                "analysis": "mixed_effects",
                "metric": metric,
                "term": term,
                "coef": float(coef),
                "stderr": float(fit.bse.get(term, np.nan)),
                "p_value": float(fit.pvalues.get(term, np.nan)),
                "formula": formula + " + (1|subject)",
                "n": int(len(df)),
            }
            for term, coef in fit.params.items()
        ]
    except Exception as exc:
        grouped = df.groupby("model")[metric].agg(["mean", "std", "count"]).reset_index()
        return [
            {
                "analysis": "fallback_group_summary",
                "reason": repr(exc),
                "model": str(row["model"]),
                "metric": metric,
                "mean": float(row["mean"]),
                "std": float(row["std"]) if np.isfinite(row["std"]) else 0.0,
                "n": int(row["count"]),
            }
            for row in grouped.to_dict("records")
        ]
