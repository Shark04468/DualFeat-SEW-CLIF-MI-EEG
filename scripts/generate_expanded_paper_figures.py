"""Generate an evidence-locked expansion of the DPC-SNN paper figures.

The script reads only completed aggregate artifacts. It does not retrain models,
rewrite results, or alter the submitted manuscript.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper_assets" / "expanded_visualizations_20260812"
FIG = OUT / "figures"
STYLE = OUT / "deepscientist-academic.mplstyle"

SOURCES = {
    "matched_curve": ROOT / "experiments/matched_continuous_controls/completed_run/MCC_8e3cb2c66043/aggregate/learning_curve.csv",
    "matched_paired": ROOT / "experiments/matched_continuous_controls/completed_run/MCC_8e3cb2c66043/aggregate/paired_bootstrap.csv",
    "matched_participant_diff": ROOT / "experiments/matched_continuous_controls/completed_run/MCC_8e3cb2c66043/aggregate/participant_paired_differences.csv",
    "soft_curve": ROOT / "experiment_result_packages/SOFTCLIF_9464f9ec7e8d_summary/aggregate/learning_curve.csv",
    "soft_paired": ROOT / "experiment_result_packages/SOFTCLIF_9464f9ec7e8d_summary/aggregate/paired_bootstrap.csv",
    "prospective_metrics": ROOT / "experiments/prospective_low_label_validation_phase_b/PLLB_PHASEB_20260810T135645Z/metrics.json",
    "prospective_model": ROOT / "experiments/prospective_low_label_validation_phase_b/PLLB_PHASEB_20260810T135645Z/analysis/participant_model_metrics_seed_averaged.csv",
    "prospective_paired": ROOT / "experiments/prospective_low_label_validation_phase_b/PLLB_PHASEB_20260810T135645Z/analysis/participant_paired_metrics.csv",
    "prospective_slopes": ROOT / "experiments/prospective_low_label_validation_phase_b/PLLB_PHASEB_20260810T135645Z/analysis/participant_gain_slopes.csv",
}

DATASET_LABEL = {
    "bci2a": "BCI2a",
    "openbmi": "OpenBMI",
    "bnci2014_004": "BCI2b",
}
DATASET_ORDER = ["bci2a", "openbmi", "bnci2014_004"]
MODEL_LABEL = {
    "ann_sew_ce": "ANN-SEW",
    "gru_ce": "GRU",
    "tcn_ce": "Causal TCN",
    "soft_clif_ce_fr0": "Soft-CLIF",
    "sew_clif_ce_fr0": "SEW-CLIF",
    "sew_clif_ce": "SEW-CLIF",
}
MODEL_COLORS = {
    "ann_sew_ce": "#6B7280",
    "gru_ce": "#0F766E",
    "tcn_ce": "#D97706",
    "soft_clif_ce_fr0": "#7C3AED",
    "sew_clif_ce_fr0": "#BE123C",
    "sew_clif_ce": "#BE123C",
}
DATASET_COLORS = {
    "bci2a": "#2563A7",
    "openbmi": "#D97706",
    "bnci2014_004": "#2F855A",
}
BUDGET_ORDER = {"n25": 0, "n50": 1, "n100": 2, "all": 3}
METRIC_LABELS = {
    "accuracy": "Accuracy",
    "balanced_accuracy": "Balanced accuracy",
    "kappa": "Cohen's kappa",
    "macro_f1": "Macro-F1",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.13, 1.07, label, transform=ax.transAxes, fontsize=11, fontweight="bold", va="top")


def save_figure(fig: plt.Figure, stem: str) -> list[Path]:
    paths = []
    for suffix, kwargs in (
        (".png", {"dpi": 320}),
        (".pdf", {}),
        (".svg", {}),
    ):
        path = FIG / f"{stem}{suffix}"
        fig.savefig(path, **kwargs)
        paths.append(path)
    plt.close(fig)
    return paths


def tick_label(row: pd.Series) -> str:
    value = float(row["examples_per_class_mean"])
    if row["budget"] == "all":
        approx = "≈" if abs(value - round(value)) > 0.05 or value > 100 else ""
        return f"{approx}{round(value):d}\n(all)"
    return f"{round(value):d}"


def figure_6_strong_controls(curve: pd.DataFrame) -> list[Path]:
    models = ["ann_sew_ce", "gru_ce", "tcn_ce", "sew_clif_ce_fr0"]
    markers = ["o", "s", "^", "D"]
    fig, axes = plt.subplots(1, 3, figsize=(10.9, 3.55), sharey=True)
    for panel, (ax, dataset) in enumerate(zip(axes, DATASET_ORDER, strict=True)):
        block = curve[curve.dataset == dataset].copy()
        for model, marker in zip(models, markers, strict=True):
            rows = block[block.model == model].sort_values("examples_per_class_mean")
            x = rows.examples_per_class_mean.to_numpy(float)
            y = 100 * rows.accuracy_mean.to_numpy(float)
            lower = 100 * rows.accuracy_bootstrap_lower.to_numpy(float)
            upper = 100 * rows.accuracy_bootstrap_upper.to_numpy(float)
            ax.errorbar(
                x,
                y,
                yerr=np.vstack((y - lower, upper - y)),
                color=MODEL_COLORS[model],
                marker=marker,
                markerfacecolor="white" if model != "sew_clif_ce_fr0" else MODEL_COLORS[model],
                markeredgewidth=1.2,
                label=MODEL_LABEL[model],
                zorder=3,
            )
        ticks = block.sort_values("examples_per_class_mean").drop_duplicates("budget")
        ax.set_xscale("log", base=2)
        ax.set_xticks(ticks.examples_per_class_mean.to_numpy(float))
        ax.set_xticklabels([tick_label(row) for _, row in ticks.iterrows()])
        ax.set_title(f"{DATASET_LABEL[dataset]}  (n={int(block.participants.max())})", fontweight="semibold")
        ax.set_xlabel("Labels per class")
        ax.set_ylim(42, 84)
        panel_label(ax, chr(ord("a") + panel))
    axes[0].set_ylabel("Participant-mean accuracy (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.03))
    fig.suptitle("Capacity-matched continuous controls on frozen ATCNet+FBCNet representations", y=1.12, fontsize=12, fontweight="semibold")
    fig.text(0.5, -0.035, "Points are participant means after seed averaging; error bars are participant-bootstrap 95% CIs.", ha="center", fontsize=8.5, color="#6B7280")
    fig.tight_layout()
    return save_figure(fig, "figure_6_strong_continuous_controls")


def figure_7_mechanism_continuum(curve: pd.DataFrame, paired: pd.DataFrame) -> list[Path]:
    models = ["ann_sew_ce", "soft_clif_ce_fr0", "sew_clif_ce_fr0"]
    budgets = [("n25", "25/class", "#2563A7", "o"), ("all", "All labels", "#D97706", "s")]
    x = np.arange(len(models), dtype=float)
    fig, axes = plt.subplots(1, 3, figsize=(10.7, 3.7), sharey=True)
    for panel, (ax, dataset) in enumerate(zip(axes, DATASET_ORDER, strict=True)):
        block = curve[curve.dataset == dataset]
        for budget, label, color, marker in budgets:
            rows = block[block.budget == budget].set_index("model").loc[models]
            y = 100 * rows.accuracy_mean.to_numpy(float)
            lower = 100 * rows.accuracy_bootstrap_lower.to_numpy(float)
            upper = 100 * rows.accuracy_bootstrap_upper.to_numpy(float)
            ax.errorbar(
                x,
                y,
                yerr=np.vstack((y - lower, upper - y)),
                color=color,
                marker=marker,
                markerfacecolor="white",
                markeredgewidth=1.2,
                label=label,
                zorder=3,
            )
        hs = paired[
            (paired.dataset == dataset)
            & (paired.budget == "n25")
            & (paired.contrast == "hard_minus_soft")
            & (paired.metric == "accuracy")
        ].iloc[0]
        ax.text(
            0.03,
            0.035,
            f"Hard−soft at 25: {100*hs.mean_difference:+.2f} pp\n95% CI [{100*hs.bootstrap_95_lower:+.2f}, {100*hs.bootstrap_95_upper:+.2f}]",
            transform=ax.transAxes,
            fontsize=8.2,
            color="#4B5563",
            va="bottom",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.86, "pad": 1.8},
        )
        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_LABEL[m] for m in models], rotation=13, ha="right")
        ax.set_title(f"{DATASET_LABEL[dataset]}  (n={int(block.participants.max())})", fontweight="semibold")
        ax.set_ylim(47, 82)
        panel_label(ax, chr(ord("a") + panel))
    axes[0].set_ylabel("Participant-mean accuracy (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Continuousized CLIF retains the low-label advantage without a detectable hard-spike increment", y=1.12, fontsize=12, fontweight="semibold")
    fig.text(0.5, -0.045, "Soft-CLIF changes only the hard threshold to a fixed sigmoid gate; all decoders use pure cross-entropy.", ha="center", fontsize=8.5, color="#6B7280")
    fig.tight_layout()
    return save_figure(fig, "figure_7_soft_clif_mechanism_continuum")


def figure_s1_effect_forest(paired: pd.DataFrame) -> list[Path]:
    models = ["ann_sew_ce", "gru_ce", "tcn_ce"]
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 4.9), sharex=True)
    for panel, (ax, model) in enumerate(zip(axes, models, strict=True)):
        rows = paired[(paired.continuous_model == model) & (paired.metric == "accuracy")].copy()
        rows["dataset_order"] = rows.dataset.map({d: i for i, d in enumerate(DATASET_ORDER)})
        rows["budget_order"] = rows.budget.map(BUDGET_ORDER)
        rows = rows.sort_values(["dataset_order", "budget_order"])
        y = np.arange(len(rows))[::-1]
        for ypos, (_, row) in zip(y, rows.iterrows(), strict=True):
            mean = 100 * float(row.mean_difference)
            lo = 100 * float(row.bootstrap_95_lower)
            hi = 100 * float(row.bootstrap_95_upper)
            ax.errorbar(mean, ypos, xerr=[[mean - lo], [hi - mean]], fmt="o", color=DATASET_COLORS[row.dataset], markerfacecolor="white", markeredgewidth=1.1, zorder=3)
        ax.axvspan(0, 25, color="#FDF2F4", zorder=0)
        ax.axvline(0, color="#4B5563", linewidth=1.0)
        labels = []
        for _, row in rows.iterrows():
            budget = row.budget.replace("n", "") if row.budget != "all" else "all"
            labels.append(f"{DATASET_LABEL[row.dataset]} · {budget}")
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.set_title(f"SEW-CLIF − {MODEL_LABEL[model]}", fontweight="semibold")
        ax.set_xlim(-3, 27)
        panel_label(ax, chr(ord("a") + panel))
    axes[1].set_xlabel("Paired accuracy difference (percentage points)")
    fig.suptitle("Participant-level paired effects against every capacity-matched continuous decoder", y=1.03, fontsize=12, fontweight="semibold")
    fig.text(0.5, -0.015, "Dots are participant-first mean differences; horizontal bars are participant-bootstrap 95% CIs.", ha="center", fontsize=8.5, color="#6B7280")
    fig.tight_layout()
    return save_figure(fig, "figure_s1_continuous_control_paired_effects")


def figure_s2_participant_distributions(diff: pd.DataFrame, paired: pd.DataFrame) -> list[Path]:
    rng = np.random.default_rng(20260812)
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.65), sharey=True)
    for panel, (ax, dataset) in enumerate(zip(axes, DATASET_ORDER, strict=True)):
        rows = diff[(diff.dataset == dataset) & (diff.continuous_model == "ann_sew_ce")].copy()
        budgets = sorted(rows.budget.unique(), key=lambda b: BUDGET_ORDER[b])
        arrays = [100 * rows[rows.budget == budget].sew_clif_minus_control_accuracy.to_numpy(float) for budget in budgets]
        positions = np.arange(len(budgets), dtype=float)
        violins = ax.violinplot(arrays, positions=positions, widths=0.72, showextrema=False)
        for body in violins["bodies"]:
            body.set_facecolor(DATASET_COLORS[dataset])
            body.set_edgecolor("none")
            body.set_alpha(0.18)
        for xpos, budget, values in zip(positions, budgets, arrays, strict=True):
            jitter = rng.uniform(-0.13, 0.13, size=len(values))
            ax.scatter(xpos + jitter, values, s=12, alpha=0.58, color=DATASET_COLORS[dataset], edgecolors="none", zorder=2)
            stat = paired[
                (paired.dataset == dataset)
                & (paired.budget == budget)
                & (paired.continuous_model == "ann_sew_ce")
                & (paired.metric == "accuracy")
            ].iloc[0]
            mean = 100 * float(stat.mean_difference)
            lo = 100 * float(stat.bootstrap_95_lower)
            hi = 100 * float(stat.bootstrap_95_upper)
            ax.errorbar(xpos, mean, yerr=[[mean - lo], [hi - mean]], fmt="D", color="#111827", markerfacecolor="white", markeredgewidth=1.0, zorder=4)
        ax.axhline(0, color="#4B5563", linewidth=1.0)
        labels = [b.replace("n", "") if b != "all" else "all" for b in budgets]
        ax.set_xticks(positions)
        ax.set_xticklabels(labels)
        ax.set_title(f"{DATASET_LABEL[dataset]}  (n={int(rows.subject.nunique())})", fontweight="semibold")
        ax.set_xlabel("Labels per class")
        panel_label(ax, chr(ord("a") + panel))
    axes[0].set_ylabel("SEW-CLIF − ANN-SEW accuracy (pp)")
    axes[0].set_ylim(-12, 31)
    fig.suptitle("Participant heterogeneity behind the objective-pure learning curves", y=1.07, fontsize=12, fontweight="semibold")
    fig.text(0.5, -0.035, "Each small point is one participant after seed averaging; diamonds and bars are paired means and 95% CIs.", ha="center", fontsize=8.5, color="#6B7280")
    fig.tight_layout()
    return save_figure(fig, "figure_s2_participant_gain_distributions")


def figure_s3_metric_robustness(paired: pd.DataFrame) -> list[Path]:
    metric_order = ["accuracy", "balanced_accuracy", "kappa", "macro_f1"]
    row_order = [(budget, metric) for budget in ("n25", "all") for metric in metric_order]
    contrast_style = {
        "soft_minus_ann": ("Soft-CLIF − ANN-SEW", "#7C3AED", "o"),
        "hard_minus_soft": ("SEW-CLIF − Soft-CLIF", "#BE123C", "s"),
    }
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 4.75), sharex=True, sharey=True)
    for panel, (ax, dataset) in enumerate(zip(axes, DATASET_ORDER, strict=True)):
        for contrast, (label, color, marker) in contrast_style.items():
            block = paired[(paired.dataset == dataset) & (paired.contrast == contrast)].set_index(["budget", "metric"])
            offset = 0.13 if contrast == "soft_minus_ann" else -0.13
            for idx, key in enumerate(row_order):
                row = block.loc[key]
                scale = 100.0
                mean = scale * float(row.mean_difference)
                lo = scale * float(row.bootstrap_95_lower)
                hi = scale * float(row.bootstrap_95_upper)
                ypos = len(row_order) - 1 - idx + offset
                ax.errorbar(mean, ypos, xerr=[[mean - lo], [hi - mean]], fmt=marker, color=color, markerfacecolor="white", markeredgewidth=1.0, label=label if idx == 0 else None, zorder=3)
        ax.axvspan(0, 25, color="#F5F3FF", alpha=0.45, zorder=0)
        ax.axvline(0, color="#4B5563", linewidth=1.0)
        ax.set_title(f"{DATASET_LABEL[dataset]}", fontweight="semibold")
        panel_label(ax, chr(ord("a") + panel))
    y = np.arange(len(row_order))[::-1]
    labels = [f"{('25' if b == 'n25' else 'all')} · {METRIC_LABELS[m]}" for b, m in row_order]
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels)
    axes[1].set_xlabel("Metric difference (percentage points; κ × 100)")
    axes[0].set_xlim(-4, 26)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Soft-CLIF conclusions are consistent across all four reported metrics", y=1.10, fontsize=12, fontweight="semibold")
    fig.text(0.5, -0.018, "All intervals use participants as the inferential unit after averaging seeds.", ha="center", fontsize=8.5, color="#6B7280")
    fig.tight_layout()
    return save_figure(fig, "figure_s3_soft_clif_metric_robustness")


def figure_8_prospective(model: pd.DataFrame, paired: pd.DataFrame, slopes: pd.DataFrame, metrics: dict) -> list[Path]:
    rng = np.random.default_rng(20260812)
    budget_order = ["n25", "n50", "all"]
    x = np.arange(3, dtype=float)
    labels = ["25", "50", "100\n(all)"]
    fig = plt.figure(figsize=(11.2, 4.10))
    grid = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.25, 0.70], wspace=0.34)
    axes = [fig.add_subplot(grid[0, i]) for i in range(3)]

    ax = axes[0]
    for model_name, offset, marker in (("ann_sew_ce", -0.055, "o"), ("sew_clif_ce", 0.055, "D")):
        block = model[model.model == model_name].copy()
        for xpos, budget in zip(x, budget_order, strict=True):
            values = 100 * block[block.budget == budget].accuracy.to_numpy(float)
            jitter = rng.uniform(-0.025, 0.025, len(values))
            ax.scatter(xpos + offset + jitter, values, s=10, alpha=0.30, color=MODEL_COLORS[model_name], edgecolors="none")
        means = [100 * block[block.budget == budget].accuracy.mean() for budget in budget_order]
        ax.plot(x + offset, means, color=MODEL_COLORS[model_name], marker=marker, label=MODEL_LABEL[model_name], zorder=4)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Accuracy (%)")
    ax.set_xlabel("Labels per class")
    ax.set_title("Frozen-decoder learning curve", fontweight="semibold")
    ax.set_ylim(45, 100)
    ax.legend(loc="lower right")
    panel_label(ax, "a")

    ax = axes[1]
    pivot = paired.pivot(index="subject", columns="budget", values="snn_minus_ann_accuracy")
    for _, row in pivot.iterrows():
        ax.plot(x, 100 * row[budget_order].to_numpy(float), color="#9CA3AF", alpha=0.48, linewidth=0.9, marker="o", markersize=2.8)
    means, lower, upper = [], [], []
    for budget in budget_order:
        stat = metrics["budget_estimates"]["accuracy"][budget]
        means.append(100 * stat["snn_minus_ann_mean"])
        lower.append(100 * stat["ci95_lower"])
        upper.append(100 * stat["ci95_upper"])
    means = np.asarray(means)
    lower = np.asarray(lower)
    upper = np.asarray(upper)
    ax.errorbar(x, means, yerr=np.vstack((means - lower, upper - means)), color="#BE123C", marker="D", linewidth=2.4, markerfacecolor="white", markeredgewidth=1.2, zorder=5, label="Participant mean (95% CI)")
    ax.axhline(0, color="#4B5563", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Labels per class")
    ax.set_ylabel("SEW-CLIF − ANN-SEW (pp)")
    ax.set_title("Paired participant gains", fontweight="semibold")
    ax.legend(loc="upper right")
    panel_label(ax, "b")

    ax = axes[2]
    values = 100 * slopes.slope_accuracy_per_log2_label.to_numpy(float)
    jitter = rng.uniform(-0.10, 0.10, len(values))
    ax.scatter(jitter, values, s=24, color="#2563A7", alpha=0.75, edgecolors="white", linewidths=0.45, zorder=3)
    slope_stat = metrics["slope_estimates"]["accuracy"]
    mean = 100 * slope_stat["mean"]
    lo = 100 * slope_stat["ci95_lower"]
    hi = 100 * slope_stat["ci95_upper"]
    ax.errorbar(0, mean, yerr=[[mean - lo], [hi - mean]], fmt="D", color="#BE123C", markerfacecolor="white", markeredgewidth=1.2, linewidth=2.2, zorder=5)
    ax.axhline(0, color="#4B5563", linewidth=1.0)
    ax.set_xlim(-0.30, 0.30)
    ax.set_xticks([0])
    ax.set_xticklabels(["12 participants"])
    ax.set_ylabel("Gain slope (pp / label doubling)")
    ax.set_title("Participant slopes", fontweight="semibold")
    panel_label(ax, "c")

    fig.suptitle("Prospectively locked BNCI2015-001 Session A → B validation", y=0.985, fontsize=12, fontweight="semibold")
    fig.text(0.5, 0.025, "Three seeds were averaged within each participant; all inference uses the 12 participants only.", ha="center", fontsize=8.5, color="#6B7280")
    fig.subplots_adjust(left=0.07, right=0.995, bottom=0.22, top=0.77, wspace=0.34)
    return save_figure(fig, "figure_8_prospective_locked_validation")


def write_manifest(output_files: list[Path]) -> None:
    source_copy_dir = OUT / "source_data"
    source_copy_dir.mkdir(parents=True, exist_ok=True)
    source_rows = []
    for key, path in SOURCES.items():
        copied = source_copy_dir / f"{key}{path.suffix}"
        shutil.copy2(path, copied)
        source_rows.append(
            {
                "source_id": key,
                "path": str(path.relative_to(ROOT)),
                "copy_path": str(copied.relative_to(ROOT)),
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
        )
    pd.DataFrame(source_rows).to_csv(OUT / "SOURCE_ARTIFACT_MANIFEST_SHA256.csv", index=False)

    output_rows = []
    for path in sorted(output_files):
        output_rows.append({"path": str(path.relative_to(ROOT)), "sha256": sha256(path), "bytes": path.stat().st_size})
    pd.DataFrame(output_rows).to_csv(OUT / "OUTPUT_MANIFEST_SHA256.csv", index=False)

    catalog = {
        "schema": "dpc-snn-figure-catalog/v1",
        "generated_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "manuscript_modified": False,
        "figures": [
            {
                "id": "Figure 6 candidate",
                "stem": "figure_6_strong_continuous_controls",
                "surface": "paper_main",
                "message": "Tests whether the original ANN control was too weak by comparing two stronger capacity-matched continuous decoders on exactly shared frozen features.",
                "suggested_section": "After the objective-pure decoder learning-curve result.",
                "status": "ready",
            },
            {
                "id": "Figure 7 candidate",
                "stem": "figure_7_soft_clif_mechanism_continuum",
                "surface": "paper_main",
                "message": "Separates continuous CLIF-like state dynamics from hard binary spike discretization under a fixed single-factor control.",
                "suggested_section": "Mechanism-control subsection following the strong continuous controls.",
                "status": "ready",
            },
            {
                "id": "Figure 8 candidate",
                "stem": "figure_8_prospective_locked_validation",
                "surface": "paper_main",
                "message": "Shows the prospectively locked, previously unaccessed Session-A to Session-B validation at the participant inferential level.",
                "suggested_section": "Dedicated prospective-validation subsection.",
                "status": "ready",
            },
            {
                "id": "Supplementary Figure S1 candidate",
                "stem": "figure_s1_continuous_control_paired_effects",
                "surface": "paper_appendix",
                "message": "Provides every participant-bootstrap accuracy contrast against ANN-SEW, GRU, and causal TCN across all budgets.",
                "status": "ready",
            },
            {
                "id": "Supplementary Figure S2 candidate",
                "stem": "figure_s2_participant_gain_distributions",
                "surface": "paper_appendix",
                "message": "Exposes participant heterogeneity behind the objective-pure learning-curve means without treating seeds as samples.",
                "status": "ready",
            },
            {
                "id": "Supplementary Figure S3 candidate",
                "stem": "figure_s3_soft_clif_metric_robustness",
                "surface": "paper_appendix",
                "message": "Checks whether the Soft-CLIF interpretation is consistent for accuracy, balanced accuracy, kappa, and macro-F1.",
                "status": "ready",
            },
        ],
    }
    catalog_path = ROOT / "paper" / "figures" / "figure_catalog.json"
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    catalog_path.write_text(json.dumps(catalog, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (OUT / "figure_catalog.json").write_text(json.dumps(catalog, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    missing = [str(path) for path in SOURCES.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing locked source artifacts:\n" + "\n".join(missing))
    plt.style.use(STYLE)
    plt.rcParams.update({"axes.titlepad": 8.0, "figure.constrained_layout.use": False})

    matched_curve = pd.read_csv(SOURCES["matched_curve"])
    matched_paired = pd.read_csv(SOURCES["matched_paired"])
    matched_diff = pd.read_csv(SOURCES["matched_participant_diff"])
    soft_curve = pd.read_csv(SOURCES["soft_curve"])
    soft_paired = pd.read_csv(SOURCES["soft_paired"])
    prospective_model = pd.read_csv(SOURCES["prospective_model"])
    prospective_paired = pd.read_csv(SOURCES["prospective_paired"])
    prospective_slopes = pd.read_csv(SOURCES["prospective_slopes"])
    prospective_metrics = json.loads(SOURCES["prospective_metrics"].read_text(encoding="utf-8"))

    output_files: list[Path] = []
    output_files += figure_6_strong_controls(matched_curve)
    output_files += figure_7_mechanism_continuum(soft_curve, soft_paired)
    output_files += figure_s1_effect_forest(matched_paired)
    output_files += figure_s2_participant_distributions(matched_diff, matched_paired)
    output_files += figure_s3_metric_robustness(soft_paired)
    output_files += figure_8_prospective(prospective_model, prospective_paired, prospective_slopes, prospective_metrics)
    write_manifest(output_files)
    print(f"Generated {len(output_files)} files under {FIG}")


if __name__ == "__main__":
    main()
