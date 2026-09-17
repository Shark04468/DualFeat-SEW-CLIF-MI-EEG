#!/usr/bin/env python3
"""Render paper-facing figures from the frozen V8 publication CSV outputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
STYLE = (
    Path.home()
    / ".codex"
    / "skills"
    / "deepscientist-figure-polish"
    / "assets"
    / "deepscientist-academic.mplstyle"
)
PAPER_RC = {
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.edgecolor": "white",
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.04,
    "axes.edgecolor": "#D8D1C7",
    "axes.labelcolor": "#4B5563",
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "axes.axisbelow": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlelocation": "left",
    "axes.titlesize": 11,
    "axes.labelsize": 10.5,
    "grid.color": "#E7E5E4",
    "grid.linewidth": 0.65,
    "grid.alpha": 0.65,
    "font.size": 10,
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Liberation Sans"],
    "xtick.color": "#6B7280",
    "ytick.color": "#6B7280",
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.frameon": False,
    "legend.fontsize": 9,
    "lines.linewidth": 1.9,
    "lines.markersize": 4.2,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
}

MODEL_LABELS = {
    "eegnet": "EEGNet",
    "fbcnet": "FBCNet",
    "atcnet": "ATCNet",
    "tcformer": "TCFormer",
    "eeg_conformer": "EEG Conformer",
    "bfatcnet": "BFATCNet",
    "atc_fbc_fusion": "ATCNet + FBCNet",
}
MODEL_ORDER = (
    "eegnet",
    "fbcnet",
    "atcnet",
    "tcformer",
    "eeg_conformer",
    "bfatcnet",
    "atc_fbc_fusion",
)
MODEL_COLORS = {
    "eegnet": "#7189A6",
    "fbcnet": "#C08A5A",
    "atcnet": "#668F80",
    "tcformer": "#8A7897",
    "eeg_conformer": "#9B6A73",
    "bfatcnet": "#7F9094",
    "atc_fbc_fusion": "#B65C52",
}
MECHANISM_MODELS = ("atcnet", "fbcnet", "atc_fbc_fusion")
FREQUENCY_ORDER = ("theta_4_8", "mu_8_13", "low_beta_13_20", "high_beta_20_30", "low_gamma_30_40")
FREQUENCY_LABELS = ("Theta\n4-8", "Mu\n8-13", "Beta-L\n13-20", "Beta-H\n20-30", "Gamma-L\n30-40")
REGION_ORDER = ("left_motor", "midline_motor", "right_motor", "frontal", "posterior")
REGION_LABELS = ("Left\nmotor", "Midline\nmotor", "Right\nmotor", "Frontal", "Posterior")
DATASET_META = {
    "bci2a": {
        "cohort": "full_9",
        "title": "BCI Competition IV-2a",
        "protocol": "Session T to E; 9 subjects; 5 seeds",
        "classes": ("Left hand", "Right hand", "Feet", "Tongue"),
    },
    "openbmi": {
        "cohort": "full_54",
        "title": "OpenBMI",
        "protocol": "Session 1 to 2; 54 subjects; 5 seeds",
        "classes": ("Left hand", "Right hand"),
    },
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def _save(figure: Any, directory: Path, stem: str) -> list[str]:
    directory.mkdir(parents=True, exist_ok=True)
    png = directory / f"{stem}.png"
    pdf = directory / f"{stem}.pdf"
    figure.savefig(png, dpi=320)
    figure.savefig(pdf)
    return [str(png), str(pdf)]


def _bootstrap_ci(values: np.ndarray, *, seed: int = 20260803) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(10_000, values.size), replace=True).mean(axis=1)
    low, high = np.percentile(samples, [2.5, 97.5])
    return float(low), float(high)


def _subject_accuracies(analysis: Path) -> dict[str, np.ndarray]:
    baseline = _read_csv(analysis / "baseline_subject_error_summary.csv")
    fusion = _read_csv(analysis / "fusion_error_subject_summary.csv")
    result: dict[str, np.ndarray] = {}
    for model in MODEL_ORDER[:-1]:
        selected = sorted(
            (row for row in baseline if row["model"] == model),
            key=lambda row: int(row["subject"]),
        )
        result[model] = np.asarray([_float(row, "accuracy_mean") for row in selected])
    selected_fusion = sorted(fusion, key=lambda row: int(row["subject"]))
    result["atc_fbc_fusion"] = np.asarray(
        [_float(row, "fusion_accuracy") for row in selected_fusion]
    )
    return result


def _accuracy_figure(analysis: Path, output: Path, dataset: str) -> dict[str, Any]:
    import matplotlib.pyplot as plt

    cohort = DATASET_META[dataset]["cohort"]
    rows = [
        row
        for row in _read_csv(analysis / "model_cohort_summary.csv")
        if row["cohort"] == cohort
    ]
    by_model = {row["model"]: row for row in rows}
    subject_values = _subject_accuracies(analysis)
    figure, axis = plt.subplots(figsize=(7.2, 3.55))
    y = np.arange(len(MODEL_ORDER))
    rng = np.random.default_rng(7)
    for index, model in enumerate(MODEL_ORDER):
        row = by_model[model]
        mean = 100.0 * _float(row, "subject_macro_accuracy")
        low = 100.0 * _float(row, "ci_low")
        high = 100.0 * _float(row, "ci_high")
        values = 100.0 * subject_values[model]
        jitter = rng.uniform(-0.095, 0.095, size=values.size)
        axis.scatter(
            values,
            np.full(values.size, index) + jitter,
            s=10 if values.size > 20 else 15,
            color=MODEL_COLORS[model],
            alpha=0.22,
            linewidth=0,
            zorder=1,
        )
        axis.plot([low, high], [index, index], color="#5F6368", linewidth=1.4, zorder=2)
        axis.scatter(
            mean,
            index,
            s=54 if model == "atc_fbc_fusion" else 42,
            color=MODEL_COLORS[model],
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )
        axis.text(high + 0.45, index, f"{mean:.1f}", va="center", fontsize=8.5)
    axis.set_yticks(y, [MODEL_LABELS[model] for model in MODEL_ORDER])
    axis.invert_yaxis()
    axis.set_xlabel("Subject-macro accuracy (%)")
    axis.set_title(
        f"{DATASET_META[dataset]['title']}: equal-exposure held-out comparison\n"
        f"{DATASET_META[dataset]['protocol']}; bars are 95% subject-bootstrap CIs",
        pad=8,
    )
    axis.grid(axis="y", visible=False)
    axis.margins(x=0.08)
    figure.tight_layout()
    exports = _save(figure, output, "accuracy_comparison")
    plt.close(figure)
    return {
        "source_data": [str(analysis / "model_cohort_summary.csv"), str(analysis / "baseline_subject_error_summary.csv"), str(analysis / "fusion_error_subject_summary.csv")],
        "exports": exports,
        "surface_class": "paper_main",
        "claim": "Equal-exposure held-out accuracy and subject heterogeneity across six baselines and the fixed fusion.",
        "revision_note": "Added subject-level points, explicit 95% CI semantics, direct mean labels, and restrained emphasis on the fixed fusion.",
    }


def _class_recall_profiles(analysis: Path) -> dict[str, np.ndarray]:
    rows = _read_csv(analysis / "fusion_error_subject_seed.csv")
    profiles: dict[str, list[np.ndarray]] = {model: [] for model in MECHANISM_MODELS}
    keys = {
        "atcnet": "atcnet_class_recall",
        "fbcnet": "fbcnet_class_recall",
        "atc_fbc_fusion": "fusion_class_recall",
    }
    subjects = sorted({int(row["subject"]) for row in rows})
    for model, key in keys.items():
        for subject in subjects:
            selected = [
                np.asarray(json.loads(row[key]), dtype=float)
                for row in rows
                if int(row["subject"]) == subject
            ]
            profiles[model].append(np.mean(selected, axis=0))
    return {model: np.asarray(values) for model, values in profiles.items()}


def _class_recall_figure(analysis: Path, output: Path, dataset: str) -> dict[str, Any]:
    import matplotlib.pyplot as plt

    profiles = _class_recall_profiles(analysis)
    classes = DATASET_META[dataset]["classes"]
    figure, axis = plt.subplots(figsize=(7.2, 3.35))
    x = np.arange(len(classes))
    offsets = (-0.18, 0.0, 0.18)
    for model, offset in zip(MECHANISM_MODELS, offsets, strict=True):
        values = 100.0 * profiles[model]
        means = values.mean(axis=0)
        lows, highs = [], []
        for class_index in range(values.shape[1]):
            low, high = _bootstrap_ci(values[:, class_index], seed=91 + class_index)
            lows.append(low)
            highs.append(high)
        errors = np.vstack((means - np.asarray(lows), np.asarray(highs) - means))
        axis.errorbar(
            x + offset,
            means,
            yerr=errors,
            fmt="o",
            label=MODEL_LABELS[model],
            color=MODEL_COLORS[model],
            markeredgecolor="white",
            markeredgewidth=0.6,
            markersize=5.5,
            linewidth=1.2,
        )
    axis.set_xticks(x, classes)
    axis.set_ylabel("Class recall (%)")
    figure.suptitle("Class-wise error complementarity (95% subject-bootstrap CI)", y=0.98)
    handles, legend_labels = axis.get_legend_handles_labels()
    figure.legend(handles, legend_labels, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 0.91))
    axis.grid(axis="x", visible=False)
    figure.subplots_adjust(top=0.72, bottom=0.17, left=0.09, right=0.99)
    exports = _save(figure, output, "fusion_class_recall")
    plt.close(figure)
    return {
        "source_data": [str(analysis / "fusion_error_subject_seed.csv")],
        "exports": exports,
        "surface_class": "paper_main",
        "claim": "ATCNet and FBCNet make different class-level errors; fixed probability fusion can balance their recalls.",
        "revision_note": "Added subject-bootstrap uncertainty and class-resolved component/fusion comparison, which was absent from the original figure set.",
    }


def _dependency_figure(analysis: Path, output: Path, dataset: str) -> dict[str, Any]:
    import matplotlib.pyplot as plt

    cohort = DATASET_META[dataset]["cohort"]
    rows = [
        row
        for row in _read_csv(analysis / "perturbation_cohort_summary.csv")
        if row["cohort"] == cohort and row["model"] in MECHANISM_MODELS
    ]
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.45))
    panels = (
        (axes[0], "frequency", FREQUENCY_ORDER, FREQUENCY_LABELS, "Frequency dependency"),
        (axes[1], "region", REGION_ORDER, REGION_LABELS, "Sensor-region dependency"),
    )
    offsets = (-0.18, 0.0, 0.18)
    for axis, kind, order, labels, title in panels:
        x = np.arange(len(order))
        for model, offset in zip(MECHANISM_MODELS, offsets, strict=True):
            selected = {
                row["name"]: row
                for row in rows
                if row["kind"] == kind and row["model"] == model
            }
            means = np.asarray([_float(selected[name], "mean_accuracy_drop_pp") for name in order])
            lows = np.asarray([_float(selected[name], "ci_low_pp") for name in order])
            highs = np.asarray([_float(selected[name], "ci_high_pp") for name in order])
            axis.errorbar(
                x + offset,
                means,
                yerr=np.vstack((means - lows, highs - means)),
                fmt="o",
                color=MODEL_COLORS[model],
                label=MODEL_LABELS[model],
                markeredgecolor="white",
                markeredgewidth=0.5,
                markersize=4.8,
                linewidth=1.0,
            )
        axis.axhline(0.0, color="#9A9A9A", linewidth=0.7)
        axis.set_xticks(x, labels)
        axis.set_title(title)
        axis.grid(axis="x", visible=False)
    axes[0].set_ylabel("Accuracy drop after occlusion (pp)")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 0.98))
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.86))
    exports = _save(figure, output, "frequency_region_dependency")
    plt.close(figure)
    return {
        "source_data": [str(analysis / "perturbation_cohort_summary.csv")],
        "exports": exports,
        "surface_class": "paper_main",
        "claim": "The fixed fusion inherits dominant mu and bilateral motor-region dependencies while combining different beta reliance from its components.",
        "revision_note": "Focused the comparison on the two fusion components and their fixed fusion, retained uncertainty, and removed the visually dense all-model heatmap from the main-paper surface.",
    }


def _subject_mechanism_figure(analysis: Path, output: Path, dataset: str) -> dict[str, Any]:
    import matplotlib.pyplot as plt

    cohort = DATASET_META[dataset]["cohort"]
    rows = [
        row
        for row in _read_csv(analysis / "fusion_error_subject_summary.csv")
        if row["cohort"] == cohort
    ]
    rows.sort(key=lambda row: _float(row, "fusion_gain_over_best_component"))
    gains = 100.0 * np.asarray([_float(row, "fusion_gain_over_best_component") for row in rows])
    double_fault = 100.0 * np.asarray([_float(row, "double_fault_rate") for row in rows])
    subjects = np.asarray([int(row["subject"]) for row in rows])
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.15), constrained_layout=True)
    colors = np.where(gains >= 0.0, "#668F80", "#B65C52")
    rank = np.arange(1, gains.size + 1)
    axes[0].vlines(rank, 0.0, gains, color=colors, linewidth=0.8, alpha=0.75)
    axes[0].scatter(rank, gains, c=colors, s=18, edgecolor="white", linewidth=0.4)
    axes[0].axhline(0.0, color="#777777", linewidth=0.8)
    axes[0].set_xlabel("Subjects ranked by fusion gain")
    axes[0].set_ylabel("Fusion gain over best component (pp)")
    axes[0].set_title(
        f"Positive in {int(np.count_nonzero(gains > 0))}/{gains.size} subjects; "
        f"median {np.median(gains):+.2f} pp"
    )
    if gains.size <= 12:
        for x_value, y_value, subject in zip(rank, gains, subjects, strict=True):
            axes[0].annotate(
                f"S{subject}",
                (x_value, y_value),
                xytext=(0, 5 if y_value >= 0 else -8),
                textcoords="offset points",
                ha="center",
                fontsize=6.5,
            )
    axes[0].grid(axis="x", visible=False)

    axes[1].scatter(
        double_fault,
        gains,
        color="#7F8F84",
        alpha=0.8,
        edgecolor="white",
        linewidth=0.5,
        s=24,
    )
    axes[1].axhline(0.0, color="#777777", linewidth=0.8)
    if double_fault.size >= 3 and not np.isclose(np.std(double_fault), 0.0):
        result = stats.spearmanr(double_fault, gains)
        slope, intercept = np.polyfit(double_fault, gains, 1)
        grid = np.linspace(double_fault.min(), double_fault.max(), 100)
        axes[1].plot(grid, intercept + slope * grid, color="#4B5563", linewidth=1.0)
        axes[1].text(
            0.04,
            0.96,
            f"Spearman rho = {result.statistic:.2f}\np = {result.pvalue:.3g}",
            transform=axes[1].transAxes,
            va="top",
            fontsize=8,
        )
    axes[1].set_xlabel("ATCNet/FBCNet double-fault rate (%)")
    axes[1].set_title("Shared errors limit fusion gain")
    axes[1].grid(alpha=0.45)
    exports = _save(figure, output, "subject_fusion_mechanism")
    plt.close(figure)
    return {
        "source_data": [str(analysis / "fusion_error_subject_summary.csv"), str(analysis / "fusion_mechanism_correlations.csv")],
        "exports": exports,
        "surface_class": "paper_main",
        "claim": "Fusion benefit is heterogeneous across subjects and is constrained by errors shared by both components.",
        "revision_note": "Replaced an unannotated disagreement scatter with sorted subject gains and a pre-specified shared-error correlation carrying rho and p values.",
    }


def _topography_figure(analysis: Path, output: Path, dataset: str) -> dict[str, Any]:
    import matplotlib.pyplot as plt
    import mne

    cohort = DATASET_META[dataset]["cohort"]
    rows = [
        row
        for row in _read_csv(analysis / "channel_saliency_cohort_summary.csv")
        if row["cohort"] == cohort and row["model"] in ("atcnet", "fbcnet")
    ]
    channel_names = list(dict.fromkeys(row["channel"] for row in rows))
    info = mne.create_info(channel_names, sfreq=250.0, ch_types="eeg")
    info.set_montage(mne.channels.make_standard_montage("standard_1020"))
    profiles = {}
    for model in ("atcnet", "fbcnet"):
        by_channel = {
            row["channel"]: _float(row, "mean_normalized_saliency")
            for row in rows
            if row["model"] == model
        }
        profiles[model] = np.asarray([by_channel[channel] for channel in channel_names])
    difference = profiles["atcnet"] - profiles["fbcnet"]
    common_min = min(float(profiles["atcnet"].min()), float(profiles["fbcnet"].min()))
    common_max = max(float(profiles["atcnet"].max()), float(profiles["fbcnet"].max()))
    difference_limit = max(float(np.max(np.abs(difference))), 1e-6)
    figure, axes = plt.subplots(1, 3, figsize=(7.2, 2.75), constrained_layout=True)
    images = []
    for axis, model in zip(axes[:2], ("atcnet", "fbcnet"), strict=True):
        image, _ = mne.viz.plot_topomap(
            profiles[model],
            info,
            axes=axis,
            show=False,
            cmap="cividis",
            vlim=(common_min, common_max),
            contours=5,
            sensors=True,
        )
        images.append(image)
        axis.set_title(MODEL_LABELS[model])
    difference_image, _ = mne.viz.plot_topomap(
        difference,
        info,
        axes=axes[2],
        show=False,
        cmap="RdBu_r",
        vlim=(-difference_limit, difference_limit),
        contours=5,
        sensors=True,
    )
    axes[2].set_title("ATCNet - FBCNet")
    first_bar = figure.colorbar(images[0], ax=axes[:2], shrink=0.72, pad=0.025)
    first_bar.set_label("Normalised saliency", fontsize=8)
    second_bar = figure.colorbar(difference_image, ax=axes[2], shrink=0.72, pad=0.025)
    second_bar.set_label("Saliency difference", fontsize=8)
    figure.suptitle("True-class gradient x input sensor dependency", fontsize=11)
    exports = _save(figure, output, "channel_topography")
    plt.close(figure)
    return {
        "source_data": [str(analysis / "channel_saliency_cohort_summary.csv")],
        "exports": exports,
        "surface_class": "appendix",
        "claim": "Both components concentrate dependency around sensorimotor electrodes, with model-specific but non-causal saliency differences.",
        "revision_note": "Added shared and signed color scales with explicit colorbars; retained a dependency rather than causal-neurophysiology interpretation.",
    }


def _render_dataset(root: Path, dataset: str) -> list[dict[str, Any]]:
    analysis = root / dataset / "analysis"
    output = analysis / "paper_figures"
    records = []
    for stem, builder in (
        ("accuracy_comparison", _accuracy_figure),
        ("fusion_class_recall", _class_recall_figure),
        ("frequency_region_dependency", _dependency_figure),
        ("subject_fusion_mechanism", _subject_mechanism_figure),
        ("channel_topography", _topography_figure),
    ):
        record = builder(analysis, output, dataset)
        records.append(
            {
                "dataset": dataset,
                "figure_id": f"{dataset}_{stem}",
                "generating_script": str(Path(__file__).resolve()),
                **record,
            }
        )
    return records


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT / "artifacts" / "experiment" / "v8_publication_baselines_d4061c11d7c4",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = _parse_args(argv)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Mirrors the bundled academic style; explicit rcParams avoid old parsers
    # treating hexadecimal colors in .mplstyle files as comments.
    plt.rcParams.update(PAPER_RC)
    catalog: list[dict[str, Any]] = []
    for dataset in DATASET_META:
        catalog.extend(_render_dataset(args.root, dataset))
    catalog_path = args.root / "figure_catalog.json"
    catalog_path.write_text(
        json.dumps(
            {
                "campaign": args.root.name,
                "frozen_results_root": str(args.root.resolve()),
                "style_source": str(STYLE),
                "figures": catalog,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"figures": len(catalog), "catalog": str(catalog_path)}, indent=2))


if __name__ == "__main__":
    main()
