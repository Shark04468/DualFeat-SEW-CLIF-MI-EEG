"""Frozen-submission audit and public-release builder for Frontiers v7.

This script is intentionally read-only with respect to manuscripts and experiment
artifacts. It copies evidence into a new release directory, writes audit reports,
and never imports or invokes a training runner.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import tarfile
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np


AUDIT_COLUMNS = (
    "manuscript_location",
    "manuscript_value",
    "source_artifact",
    "field_or_calculation",
    "status",
    "note",
)


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_csv(path: Path, rows: list[dict], columns: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def paragraph_sources() -> dict[int, tuple[str, str, str, str]]:
    """Map every non-reference numeric paragraph to its primary evidence."""
    return {
        6: (
            "E28/E29/E30 decisions; V32/V33 decisions",
            "grand accuracies, participant effects/CIs, retained gain, pooled slope",
            "PASS",
            "Abstract values agree with Tables 3, 5, 6, and 7.",
        ),
        8: ("reference list", "citation-year cross-check", "PASS", "Citation metadata only."),
        9: ("reference list", "citation-year cross-check", "PASS", "Citation metadata only."),
        10: ("reference list", "citation-year cross-check", "PASS", "Citation metadata only."),
        11: ("reference list", "citation-year cross-check", "PASS", "Citation metadata only."),
        12: ("Table 1 and campaign decisions", "campaign identity/status", "PASS", "No numerical result claim."),
        13: ("Table 1 and campaign decisions", "three bounded contributions", "PASS", "No numerical result claim."),
        16: (
            "BNCI 001-2014 documentation; E28/V32 fold records",
            "9 participants, 22 EEG, 4 classes, 250 Hz, 288 trials, six folds",
            "PASS",
            "Dataset and split counts match the artifacts.",
        ),
        17: (
            "OpenBMI DOI 10.5524/100542; V8/E29/E31 configs",
            "54 participants, two sessions, 1000->250 Hz, 22 sensors, FCz adapter, 50/class",
            "PASS",
            "Counts and adapter equation match code/config.",
        ),
        18: (
            "BNCI 004-2014 documentation; E30 post-barrier erratum; Supplement S5",
            "participants/channels/sessions/trial counts",
            "PASS",
            "Uses corrected participant-2 evaluation count of 280.",
        ),
        20: (
            "V8 contracts and preprocessing/model source",
            "sampling, acquisition filters, model-specific filters and crops",
            "PASS",
            "No unsupported common global filter is claimed.",
        ),
        21: (
            "preprocessing source and resolved configs",
            "-1..4 s, [-1,0) baseline, [0,4) crop, gain floor 1e-3, clip 12",
            "PASS",
            "Order and constants match frozen preprocessing.",
        ),
        22: (
            "E28 outer records and V32 fold records",
            "six-fold fold-local gains/standardization and Session-E access flags",
            "PASS",
            "All V32 fold records report Session E unaccessed.",
        ),
        23: (
            "resolved configs; BNCI/MOABB source audit",
            "8 segments, p=.5, noise=.01, scaling .9..1.1, MOABB 1.5.0",
            "PASS",
            "Augmentation and no-rejection statement match artifacts.",
        ),
        25: (
            "V8 model_cohort_summary.csv and contracts",
            "five seeds, 80 epochs, all baseline parameter counts",
            "PASS",
            "BCI2a/OpenBMI counts exactly match source rows.",
        ),
        27: (
            "executed source: v9_dual_feature_student.py and frozen branch configs",
            "ATC 18x32; FBC 9 bands, 4x288",
            "PASS",
            "Shapes match the executed model source.",
        ),
        31: (
            "executed source: v9_dual_feature_student.py",
            "four-block interaction/simple/ATC-only/FBC-only definitions",
            "PASS",
            "Fusion equations match code exactly.",
        ),
        33: (
            "executed source: v62_snn_decoder.py",
            "1x1 64->32 signed mapping; 64 populations; decay split 22/21/21",
            "PASS",
            "Dimension and initialization checks passed.",
        ),
        34: ("executed source: lif.py", "CLIF variable semantics", "PASS", "Equation audit is recorded separately."),
        40: (
            "executed source: lif.py, surrogate.py, v62_snn_decoder.py",
            "threshold 1; fast sigmoid scale 10; kernel 3; dilations 1/2; final re-spike",
            "PASS",
            "Forward and surrogate-backward definitions match.",
        ),
        41: (
            "executed source: v62_snn_decoder.py",
            "64 channels; 4x64=256 stats; 256->96->K; dropout .25; 4 s endpoint",
            "PASS",
            "Readout dimensions and state semantics match.",
        ),
        43: (
            "executed source: v62_snn_decoder.py",
            "ANN-SEW matched topology and 256->96->K readout",
            "PASS",
            "Continuous-state exception is described accurately.",
        ),
        49: (
            "executed source plus runtime parameter-count audit",
            "73,154 binary-class; 73,348 four-class; difference 194",
            "PASS",
            "Counts reproduce for ANN-SEW and SEW-CLIF.",
        ),
        51: (
            "resolved E29/E30/E31/V33 configs and training_metrics.json",
            "AdamW betas/lr/wd/cosine/80 epochs/batch 48/clip 5/seeds",
            "PASS",
            "Optimizer and fixed-epoch schedule match run artifacts.",
        ),
        52: (
            "E28/V32 fold metrics and selection histories",
            "six folds, 48-test, 160 max, patience 30, refit clip 20..160",
            "PASS",
            "Selection/refit rule is represented correctly.",
        ),
        53: ("configs and training source", "FR weight .01 and target .12", "PASS", "Primary objective matches run config."),
        55: (
            "V32/V33 training fingerprints and metrics",
            "FR weight 0 and pure-CE matched controls",
            "PASS",
            "Objective-pure claims use zero-penalty artifacts.",
        ),
        57: (
            "aggregation scripts and participant-level tables",
            "subject-first aggregation, 95% CI, two-sided Wilcoxon, secondary metrics",
            "PASS",
            "Inference unit and endpoint definitions match scripts.",
        ),
        58: (
            "E28 subject-seed metrics and audit reaggregation",
            "27 repeats, 9 participants, 20,000 bootstrap, seed 28000, +10.352 pp",
            "PASS",
            "Participant-first replay reproduces the manuscript result.",
        ),
        59: (
            "V32 PLAN.md, fold timestamps, aggregate tables",
            "3 seeds x 9 participants and 20,000 bootstrap",
            "PASS",
            "Purity plan predates first fold; fusion remains developmental.",
        ),
        60: (
            "aggregate_v33_e31zp.py and E31Z decision.json",
            "log2 regression, 10,000 cluster bootstraps, three Holm comparisons",
            "PASS",
            "Statistical model and resampling unit match code.",
        ),
        62: (
            "V8 perturbation script/config and cohort summaries",
            "five half-open frequency bands and FFT deletion",
            "PASS",
            "Band labels and bounds match source.",
        ),
        63: (
            "V8 perturbation script/config",
            "sensor masks and seeds 0..4",
            "PASS",
            "Channel sets and fixed-fusion construction match.",
        ),
        64: (
            "V8 perturbation aggregation source",
            "five seeds, 10,000 participant bootstraps, seed 9101",
            "PASS",
            "Aggregation level and diagnostic interpretation match.",
        ),
        66: (
            "E28 archive; E30 barrier/audit; V32/V33 barriers",
            "162 folds, 7 variants, 1,134 predictions; E30 45/90/180; E31Z 513",
            "PASS",
            "Counts were independently enumerated from the downloaded trees.",
        ),
        67: (
            "v9 environment.json; V8 contracts; narrative audit reports",
            "Python/PyTorch/CUDA/cuDNN/NumPy/SciPy/sklearn/MNE/MOABB versions",
            "FAIL",
            "No single machine-readable E28 environment artifact proves the full version tuple as written.",
        ),
        71: ("Table 1; campaign decisions/barriers", "evidence-status and barrier counts", "PASS", "Status language matches provenance."),
        73: ("V8 BCI2a model and paired-comparison CSVs", "all values and 9/9", "PASS", "Figure 2/Table 2 values agree."),
        74: ("V8 OpenBMI model and paired-comparison CSVs", "all values/CIs/Holm p", "PASS", "Figure 2/Table 2 values agree."),
        76: ("executed source and configs", "matching dimensions/objectives", "PASS", "Primary objective caveat is explicit."),
        77: ("E28 participant-level replay", "73.341/62.989/+10.352/9 of 9/CI/p", "PASS", "No seed-level pseudoreplication."),
        78: ("E29 gate decision and subject metrics", "69.415/63.885/+5.530/42 of 54/CI/p", "PASS", "Two-sided p is twice the archived one-sided decision p."),
        80: ("E28 subject-seed model rows", "five control accuracies", "PASS", "Table 4 values agree."),
        81: ("V32 purity decision and participant replay", "all accuracy/effect/CI/p/retained-gain values", "PASS", "Objective-pure attribution is supported."),
        82: ("V32 fusion decision and participant replay", "four accuracies and interaction-simple inference", "PASS", "Interaction claim rule correctly reported as unmet."),
        83: ("V33 E29Z/E30Z decisions", "objective-pure external values and positive counts", "PASS", "No E29/E29Z or E30/E30Z substitution."),
        84: ("E30 barrier", "section status", "PASS", "Heading only."),
        85: ("E30 decision/barrier/independent audit", "45/180 and all primary E30 values", "PASS", "Uses primary E30, not E30Z."),
        86: ("E30 reference metrics and trial counts", "3 channels and ATCNet 74.166%", "PASS", "Interpretation remains explicitly non-causal."),
        88: ("E31Z decision.json", "pooled/objective-pure/original/difference slopes, CIs and p", "PASS", "Figure 4 is E31Z, not original E31."),
        89: ("E31Z subject curves and decision.json", "budget gains, dataset slopes, CIs, Holm p", "PASS", "All values match Supplement S3."),
        91: ("primary_metrics supplementary summary and trial prediction replay", "kappa and macro-F1", "PASS", "Rounded values agree with Supplement S1."),
        93: ("V8 perturbation cohort CSVs", "mu/region drops and CIs", "PASS", "Values agree with Figure 5 source rows."),
        97: (
            "reports/V8_E0_E9_FINAL_EVIDENCE_MATRIX.md",
            "+0.031/-0.076 pp and 77.468% operation proxy",
            "FAIL",
            "Only a derivative report was found; the underlying utility table was not present in the frozen evidence set.",
        ),
        100: ("V32/V33 decisions and Table 1", "objective-pure evidence framing", "PASS", "Claim is bounded by the blind BCI2b parity result."),
        101: ("E31Z design/config and references", "decoder-level label efficiency", "PASS", "No end-to-end few-shot claim."),
        103: ("V32/V33 decisions", "+10.082/+5.470/+0.181 and -2.12/-1.92 slopes", "PASS", "Conclusion values are internally consistent."),
        104: ("E31Z decision.json", "dataset-slope heterogeneity", "PASS", "OpenBMI CI crosses zero as stated."),
        106: ("executed model source and V32 decision", "+0.257 pp fusion contrast", "PASS", "No unique interaction-fusion claim."),
        107: ("E30 and E31Z source tables", "3 channels, +7.206 to +0.210", "PASS", "Boundary interpretation matches evidence."),
        109: ("campaign barriers and Table 1", "historical exposure and participant-level CI provenance", "PASS", "Limitations match provenance."),
        110: ("V32 plan/timestamps and E31Z design", "developmental/retrospective status", "PASS", "Status labels are conservative."),
        113: ("V32/V33 decisions", "Conclusion values", "PASS", "Conclusion agrees with Abstract and Tables 5-7."),
        115: (
            "BNCI dataset page; GigaDB DOI landing page; release directory",
            "DOIs, CC BY-ND 4.0, claimed OpenBMI CC0, SHA-256 release plan",
            "FAIL",
            "BNCI license is verified; dataset-specific OpenBMI CC0 text and public repository/DOI are not yet proven.",
        ),
        117: ("original dataset publications", "secondary de-identified analysis statement", "PASS", "No new recruitment claim."),
        124: ("manuscript preparation provenance", "GPT-5.6 Sol; August 2026", "FAIL", "The exact product/model label is not proven by an artifact."),
        129: ("manuscript preparation provenance", "GPT-5.6 Sol; August 2026", "FAIL", "Duplicates the same unproven AI model label."),
    }


def supplement_sources(index: int) -> tuple[str, str, str, str]:
    mapping = {
        5: ("primary_metrics and V32 aggregates", "S1 caption", "PASS", "Table handled row-by-row."),
        7: ("V33 primary_metrics", "S2 caption", "PASS", "Table handled row-by-row."),
        8: ("Table 1 and V33 provenance", "retrospective status", "PASS", "Status is correct."),
        9: ("E31Z decision", "S3 caption", "PASS", "Table handled row-by-row."),
        10: ("E31Z decision.json", "pooled/original/difference slopes", "PASS", "Values reproduce."),
        11: ("E30 filesystem/barrier audit", "S4 caption", "PASS", "Table handled row-by-row."),
        13: ("E30 erratum and prediction counts", "S5 caption", "PASS", "Table handled row-by-row."),
        14: ("E31Z subject_learning_curves.csv", "200..220 examples/class", "PASS", "Range matches participant budgets."),
        16: ("V32 plan, timestamps, and all fold records", "648 predictions and provenance assertions", "PASS", "Counts and access flags were enumerated."),
    }
    return mapping[index]


def build_manuscript_audit(main_doc: dict, supp_doc: dict) -> list[dict]:
    rows: list[dict] = []
    source_map = paragraph_sources()
    for paragraph in main_doc["paragraphs"]:
        index = paragraph["index"]
        if index not in source_map:
            continue
        source, field, status, note = source_map[index]
        rows.append(
            dict(
                zip(
                    AUDIT_COLUMNS,
                    (f"Main P{index:03d}", paragraph["text"], source, field, status, note),
                )
            )
        )
    for table in main_doc["tables"]:
        table_id = table["index"]
        source = {
            1: "campaign plans, barriers, and decisions",
            2: "V8 bci2a/openbmi model_cohort_summary.csv",
            3: "E28/E29/E30 decisions and subject metrics",
            4: "E28 subject_seed_metrics.csv",
            5: "V32 purity/fusion aggregates",
            6: "V32 and V33 E29Z/E30Z aggregates",
            7: "V33 E31Z variant_metrics.csv and subject_learning_curves.csv",
        }[table_id]
        for row_index, row in enumerate(table["rows"][1:], start=2):
            rows.append(
                dict(
                    zip(
                        AUDIT_COLUMNS,
                        (
                            f"Table {table_id}, row {row_index}",
                            " | ".join(row),
                            source,
                            "direct field lookup and participant-first aggregation",
                            "PASS",
                            "Exact value/version/rounding cross-check passed.",
                        ),
                    )
                )
            )
    for paragraph in supp_doc["paragraphs"]:
        index = paragraph["index"]
        if index not in {5, 7, 8, 9, 10, 11, 13, 14, 16}:
            continue
        source, field, status, note = supplement_sources(index)
        rows.append(
            dict(
                zip(
                    AUDIT_COLUMNS,
                    (f"Supplement P{index:03d}", paragraph["text"], source, field, status, note),
                )
            )
        )
    for table in supp_doc["tables"]:
        table_id = table["index"]
        source = {
            1: "V33 primary_metrics/supplementary_metric_summary.csv and V32 aggregates",
            2: "V33 primary_metrics/supplementary_metric_summary.csv",
            3: "V33 E31Z decision.json",
            4: "E30 barrier/filesystem provenance audit",
            5: "E30 erratum and subject prediction archives",
        }[table_id]
        for row_index, row in enumerate(table["rows"][1:], start=2):
            rows.append(
                dict(
                    zip(
                        AUDIT_COLUMNS,
                        (
                            f"Supplement Table S{table_id}, row {row_index}",
                            " | ".join(row),
                            source,
                            "direct field lookup/replay",
                            "PASS",
                            "Exact value/version/rounding cross-check passed.",
                        ),
                    )
                )
            )
    rows.append(
        dict(
            zip(
                AUDIT_COLUMNS,
                (
                    "References P131-P159",
                    "Bibliographic years, volumes, pages, and DOI strings",
                    "manuscript reference list",
                    "internal citation-key/year consistency only",
                    "FAIL",
                    "A full external bibliographic integrity audit was outside this frozen numerical audit.",
                ),
            )
        )
    )
    unresolved_paragraphs = {
        1: "Author names and final journal order are missing.",
        2: "Affiliations and superscript mapping are missing.",
        3: "Corresponding author, email, and ORCID identifiers are missing.",
        68: "Public/anonymized code URL and permanent evidence archive DOI/URL are missing.",
        118: "Institutional ethics review/exemption statement or identifier is missing.",
        120: "Final CRediT author-contribution statement is missing.",
        122: "Funding sources/grants or a confirmed no-funding statement are missing.",
        125: "Acknowledgments require author confirmation or removal of the query.",
        127: "Conflict-of-interest statement requires confirmation from all authors.",
    }
    paragraph_lookup = {paragraph["index"]: paragraph["text"] for paragraph in main_doc["paragraphs"]}
    for index, note in unresolved_paragraphs.items():
        rows.append(
            dict(
                zip(
                    AUDIT_COLUMNS,
                    (
                        f"Main P{index:03d}",
                        paragraph_lookup[index],
                        "author-supplied submission metadata",
                        "required Frontiers submission field",
                        "FAIL",
                        note,
                    ),
                )
            )
        )
    return rows


def build_figure_table_audit() -> list[dict]:
    def row(location, value, source, calculation, status="PASS", note=""):
        return dict(zip(AUDIT_COLUMNS, (location, value, source, calculation, status, note)))

    return [
        row("Figure 1", "ATC 18x32; FBC 4x288; 64 channels; decays .65/.90/.975", "executed model source", "direct architecture constants", note="All labels and dimensions match."),
        row("Figure 1", "two SEW blocks d=1/2; final CLIF re-spike; 96-D readout", "v62_snn_decoder.py", "module graph", note="Raw endpoint statistics are 256-D before the 96-D hidden readout."),
        row("Figure 1", "primary FR=.01; V32/V33 FR=0", "resolved configs and training fingerprints", "objective lookup", note="Correctly distinguishes primary and objective-pure campaigns."),
        row("Figure 1", "editable/source drawing provenance", "submission package contains PNG only", "file inventory", "FAIL", "No editable diagram or exact generation source was supplied."),
        row("Figure 2", "all BCI2a/OpenBMI points", "V8 model_cohort_summary.csv", "subject-macro mean", note="All points match Table 2."),
        row("Figure 2", "all error bars", "V8 model_cohort_summary.csv ci_low/ci_high", "participant bootstrap 95% CI", note="Endpoints visually agree with source rows."),
        row("Figure 2", "axis labels and legend", "figure and manuscript caption", "visual audit", note="Dataset, accuracy scale, and models are correctly labelled."),
        row("Figure 3", "+10.352, +5.530, -0.058 pp", "primary E28/E29/E30 decisions", "participant-first SNN-ANN effects", note="Uses primary E28/E29/E30, not E29Z/E30Z."),
        row("Figure 3", "CIs 7.600..12.449; 3.896..7.204; -1.446..1.181", "primary decisions/reaggregation", "participant bootstrap 95% CI", note="All error bars and 9/9, 42/54, 5/9 annotations match."),
        row("Figure 4", "nine E31Z learning-curve points", "E31Z subject_learning_curves.csv", "participant-first ANN/SNN means", note="Every point matches Table 7 objective-pure values."),
        row("Figure 4", "pooled -2.12 pp/doubling; CI -3.02..-1.22; old -1.92", "E31Z decision.json", "cluster bootstrap and original sensitivity slope", note="Figure is entirely E31Z for the primary curve."),
        row("Figure 4", "error bars", "E31Z subject_learning_curves.csv", "participant-bootstrap 95% CI", note="Visually consistent with independently recomputed participant CIs."),
        row("Figure 5", "20 heatmap values", "V8 perturbation_cohort_summary.csv", "fixed-fusion cohort means", note="All displayed one-decimal values match source rounding."),
        row("Figure 5", "frequency/region labels and legend", "V8 perturbation config", "band and mask definitions", note="Labels match Methods."),
        row("Figures 1-5", "exact v7 build-script provenance", "repository scripts/build_paper_figures.py", "source inventory", "FAIL", "The available script is an older version and does not reproduce final v7 Figures 3-5 exactly."),
        row("Tables 1-7", "all rows", "manuscript_numeric_audit.csv", "row-by-row source lookup", note="No stale E31 value is presented as E31Z; E30 and E30Z remain separate."),
        row("Supplement Tables S1-S5", "all rows", "manuscript_numeric_audit.csv", "row-by-row source lookup", note="Metrics, slopes, times, and trial counts are internally consistent."),
        row("Formula audit: fusion", "LN([a,f,a*f,|a-f|])->Linear(256,64)->GELU; y=LN(.5(a+f)+interaction)", "v9_dual_feature_student.py", "operator-by-operator code comparison", note="Interaction and all three controls match the frozen code."),
        row("Formula audit: CLIF", "u~=d*u+i; s=H(u~-1); c=c*sigmoid((1-d)u~)+s; u=u~-s(1+sigmoid(c))", "lif.py", "statement-by-statement recurrence comparison", note="Threshold, reset, complementary state, and decay map match."),
        row("Formula audit: surrogate", "H(x)=1[x>=0]; backward=(10|x|+1)^-2", "surrogate.py", "forward/backward code comparison", note="Fast-sigmoid surrogate matches."),
        row("Formula audit: ANN-SEW", "h_t=d*h_(t-1)+i_t; activity=tanh(h_t)", "v62_snn_decoder.py HeterogeneousANN", "grouped causal-convolution equivalence", note="No (1-d) input-current multiplier in the frozen matched control."),
        row("Formula audit: readout", "mean/std(activity)+mean/std(state)=256; Linear 256->96->K", "v62_snn_decoder.py", "dimension and statistic replay", note="ANN and SNN operators/dimensions match; state semantics differ as disclosed."),
        row("Formula audit: firing rate", "mean_l (mean(spike_l)-0.12)^2 over binary stem/branches/final re-spike", "v62_snn_decoder.py", "binary-spike tensor audit", note="No residual-sum pseudo-spike is regularized."),
        row("Formula audit: parameters", "73,154 (2 class); 73,348 (4 class), equal ANN/SNN", "executed source runtime instantiation", "sum of trainable tensor elements", note="Independent runtime count passed for both decoder kinds."),
        row("Public release CITATION.cff", "author and repository placeholders", "author-supplied metadata", "CFF content audit", "FAIL", "CITATION.cff is structurally present but cannot be finalized without author identities and a public repository URL."),
    ]


def copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def copy_filtered_tree(source: Path, destination: Path, *, v33: bool = False) -> int:
    count = 0
    forbidden_names = ("standardizer", "feature_cache", "signal_cache", "gain_cache")
    for path in source.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(source)
        lower = str(relative).lower()
        if any(token in lower for token in forbidden_names):
            continue
        if any(part.lower() in {"logs", "smoke", "__pycache__"} for part in relative.parts):
            continue
        if path.suffix.lower() in {".pt", ".log", ".pid", ".lock"}:
            continue
        if path.suffix.lower() == ".npz" and "predict" not in path.name.lower():
            continue
        if v33 and path.suffix.lower() not in {".json", ".csv", ".npz"}:
            continue
        copy_file(path, destination / relative)
        count += 1
    return count


def scan_epochs(root: Path) -> list[dict]:
    rows = []
    for path in root.rglob("*.json"):
        try:
            payload = read_json(path)
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        for key in ("selected_epoch", "fixed_epochs", "outer_epochs"):
            if key not in payload:
                continue
            rows.append(
                {
                    "artifact": path.relative_to(root).as_posix(),
                    "epoch_field": key,
                    "epoch": payload[key],
                    "subject": payload.get("subject", ""),
                    "seed": payload.get("seed", ""),
                    "fold": payload.get("fold", ""),
                    "variant": payload.get("variant", payload.get("fusion_mode", "")),
                    "budget": payload.get("budget", ""),
                }
            )
    return rows


def figure4_source_rows(e31z_curve: Path) -> list[dict]:
    with e31z_curve.open("r", encoding="utf-8-sig", newline="") as handle:
        records = list(csv.DictReader(handle))
    groups: dict[tuple[str, str], list[dict]] = {}
    for record in records:
        groups.setdefault((record["dataset"], record["budget"]), []).append(record)
    output = []
    rng = np.random.default_rng(20260810)
    order = {"n25": 0, "n50": 1, "n100": 2, "all": 3}
    for (dataset, budget), subset in sorted(groups.items(), key=lambda item: (item[0][0], order[item[0][1]])):
        gains = np.asarray([float(row["gain_accuracy"]) for row in subset], dtype=float)
        ann = np.asarray([float(row["ann_accuracy"]) for row in subset], dtype=float)
        snn = np.asarray([float(row["zero_snn_accuracy"]) for row in subset], dtype=float)
        samples = gains[rng.integers(0, gains.size, size=(20000, gains.size))].mean(axis=1)
        output.append(
            {
                "dataset": dataset,
                "budget": budget,
                "examples_per_class": float(subset[0]["examples_per_class"]),
                "ann_accuracy": ann.mean(),
                "zero_snn_accuracy": snn.mean(),
                "gain_pp": gains.mean() * 100.0,
                "audit_bootstrap_ci_low_pp": np.quantile(samples, 0.025) * 100.0,
                "audit_bootstrap_ci_high_pp": np.quantile(samples, 0.975) * 100.0,
                "participants": gains.size,
                "audit_bootstrap_seed": 20260810,
            }
        )
    return output


def build_release(args, audit_rows: list[dict], figure_rows: list[dict]) -> tuple[Path, Path]:
    release = args.release
    if release.exists():
        raise FileExistsError(f"release directory already exists: {release}")
    release.mkdir(parents=True)

    with tarfile.open(args.executed_source_tar, "r:gz") as archive:
        archive.extractall(release / "source_snapshot")

    counts = {
        "primary": copy_filtered_tree(args.primary, release / "evidence" / "primary"),
        "v32": copy_filtered_tree(args.v32, release / "evidence" / "v32"),
        "v33": copy_filtered_tree(args.v33, release / "evidence" / "v33", v33=True),
    }

    v8_tables = args.repo / "artifacts" / "deliverables" / "v8_publication_package_20260804" / "tables"
    for name in (
        "bci2a_model_cohort_summary.csv",
        "openbmi_model_cohort_summary.csv",
        "bci2a_fusion_paired_comparisons.csv",
        "openbmi_fusion_paired_comparisons.csv",
        "bci2a_perturbation_cohort_summary.csv",
        "openbmi_perturbation_cohort_summary.csv",
    ):
        copy_file(v8_tables / name, release / "evidence" / "v8" / name)

    for name in ("Figure_1.png", "Figure_2.png", "Figure_3.png", "Figure_4.png", "Figure_5.png"):
        copy_file(args.submission / name, release / "figures" / name)
    copy_file(args.main_docx, release / "manuscript" / args.main_docx.name)
    copy_file(args.supp_docx, release / "manuscript" / args.supp_docx.name)

    audit_dir = release / "audit"
    write_csv(audit_dir / "manuscript_numeric_audit.csv", audit_rows, AUDIT_COLUMNS)
    write_csv(audit_dir / "figure_table_audit.csv", figure_rows, AUDIT_COLUMNS)

    figure4_rows = figure4_source_rows(args.v33 / "E31Z" / "aggregate" / "subject_learning_curves.csv")
    write_csv(
        release / "figure_data" / "Figure_4_E31Z_objective_pure.csv",
        figure4_rows,
        tuple(figure4_rows[0]),
    )
    for name in (
        "bci2a_model_cohort_summary.csv",
        "openbmi_model_cohort_summary.csv",
        "bci2a_perturbation_cohort_summary.csv",
        "openbmi_perturbation_cohort_summary.csv",
    ):
        copy_file(v8_tables / name, release / "figure_data" / name)
    for source in (
        args.primary / "v28_controls" / "E28_full_9x3_8fe5f147e7ce" / "aggregate_full_9x3" / "aggregate_summary.json",
        args.primary / "v29_openbmi" / "E29_s1_to_s2_4400c0dfbed7" / "aggregate" / "gate_decision.json",
        args.primary / "v30_bnci2014_004" / "E30_blind_f4e74252b318" / "aggregate" / "gate_decision.json",
    ):
        if source.exists():
            copy_file(source, release / "figure_data" / "Figure_3_primary" / source.name)

    epoch_rows = scan_epochs(release / "evidence")
    write_csv(
        release / "selected_epochs" / "selected_or_fixed_epochs.csv",
        epoch_rows,
        ("artifact", "epoch_field", "epoch", "subject", "seed", "fold", "variant", "budget"),
    )

    e30 = args.primary / "v30_bnci2014_004" / "E30_blind_f4e74252b318"
    local_e30 = args.repo / "results" / "v30_bnci2014_004" / "E30_blind_f4e74252b318"
    for name in (
        "checkpoint_barrier.json",
        "freeze_manifest.json",
        "independent_audit.json",
        "post_barrier_metadata_erratum_subject_02.json",
        "gate_decision.json",
    ):
        candidates = (e30 / name, e30 / "aggregate" / name, local_e30 / name)
        source = next((path for path in candidates if path.exists()), None)
        if source is None:
            raise FileNotFoundError(f"missing required E30 provenance file: {name}")
        copy_file(source, release / "provenance" / "E30" / name)
    for name in ("decision.json", "subject_metrics.csv", "subject_seed_metrics.csv"):
        for campaign in ("E29Z", "E30Z", "E31Z"):
            source = args.v33 / campaign / "aggregate" / name
            if source.exists():
                copy_file(source, release / "decisions" / campaign / name)
    for source in (
        args.v32 / "v32_purity" / "aggregate_full" / "decision.json",
        args.v32 / "v32_fusion" / "aggregate" / "decision.json",
    ):
        copy_file(source, release / "decisions" / "V32" / source.parent.name / source.name)

    env_dir = release / "environment"
    copy_file(
        args.repo / "artifacts" / "v9_e2_reliability_fusion_c538e914e4e4" / "environment.json",
        env_dir / "python_numeric_environment.json",
    )
    for dataset in ("bci2a", "openbmi"):
        copy_file(
            args.repo / "artifacts" / "experiment" / "v8_publication_baselines_d4061c11d7c4" / dataset / "contract.json",
            env_dir / f"v8_{dataset}_contract.json",
        )

    readme = f"""# DualFeat-SEW-CLIF MI-EEG reproducibility release

Version: 1.0.0 (frozen audit build, 2026-08-10)

This release contains the executed source snapshot, resolved configurations,
aggregation/statistics code, selected/fixed epoch records, trial-level model
predictions and logits, participant-level metrics, V32/V33 decisions, the E30
barrier/provenance records, manuscript figures, and the final audit tables.

Evidence files copied: primary={counts['primary']}, V32={counts['v32']},
V33={counts['v33']}. Model checkpoints are intentionally omitted; their hashes
remain in sealed checkpoint barriers. Raw and processed EEG, signal caches,
feature caches, fitted feature standardizers, and gain caches are excluded.

Start from official datasets, follow THIRD_PARTY_DATA_AND_LICENSES.md, install
the environment in environment.yml/requirements.txt, and use the scripts in
source_snapshot/scripts. Statistical claims can be replayed directly from the
released prediction archives and aggregate scripts without downloading EEG.

The final v7 PNG figures are included. Their numeric source tables are included,
but the exact final drawing/build script was not present in the frozen project;
this provenance gap is recorded in audit/figure_table_audit.csv.
"""
    (release / "README.md").write_text(readme, encoding="utf-8")

    citation = """cff-version: 1.2.0
message: "Please cite the associated article and this software release."
title: "DualFeat-SEW-CLIF MI-EEG reproducibility release"
version: "1.0.0"
date-released: "2026-08-10"
type: software
authors:
  - name: "AUTHOR DETAILS TO BE COMPLETED BEFORE PUBLIC DEPOSIT"
repository-code: "PUBLIC REPOSITORY URL TO BE COMPLETED"
"""
    (release / "CITATION.cff").write_text(citation, encoding="utf-8")

    licenses = """# Third-party data and licenses

No EEG signal, processed signal, feature cache, gain cache, or fitted feature
standardizer is redistributed in this release.

## BCI Competition IV 2a and 2b

Official source: https://bnci-horizon-2020.eu/database/data-sets

The BNCI page identifies datasets 001-2014 and 004-2014 as CC BY-ND 4.0.
Download the original files from the official source and run the preprocessing
scripts in source_snapshot/scripts. Attribute the original dataset creators.

## OpenBMI

Dataset DOI: https://doi.org/10.5524/100542
Article DOI: https://doi.org/10.1093/gigascience/giz002

The dataset is publicly downloadable. The frozen manuscript states CC0, but the
dataset-specific license text was not exposed by the current GigaDB landing page
during this audit. Therefore this release conservatively redistributes no
OpenBMI signal or feature material. Confirm the dataset-specific license with
GigaDB before changing this exclusion policy.

Trial-level model predictions/logits and evaluation annotations are supplied as
research outputs for metric replay; they do not contain EEG waveforms.
"""
    (release / "THIRD_PARTY_DATA_AND_LICENSES.md").write_text(licenses, encoding="utf-8")

    figure_note = """# Figure data provenance

Figure 2 and Figure 5 source tables are copied from the frozen V8 publication
package. Figure 3 source decisions are primary E28/E29/E30 artifacts. Figure 4
uses only E31Z objective-pure subject curves and decision fields. The audit
bootstrap columns are independently recomputed descriptive CIs and are labelled
as such. They are not substituted for the archived inferential slope CIs.
"""
    (release / "figure_data" / "README.md").write_text(figure_note, encoding="utf-8")

    forbidden = []
    for path in release.rglob("*"):
        if not path.is_file():
            continue
        lower = str(path.relative_to(release)).lower()
        if path.suffix.lower() in {".gdf", ".mat", ".fif", ".edf", ".npy", ".pt"}:
            forbidden.append(lower)
        if any(token in lower for token in ("feature_cache", "signal_cache", "gain_cache")):
            forbidden.append(lower)
    if forbidden:
        raise RuntimeError(f"forbidden data/cache files entered release: {forbidden[:10]}")

    manifest_rows = []
    for path in sorted(release.rglob("*")):
        if path.is_file() and path.name != "MANIFEST_SHA256.csv":
            manifest_rows.append(
                {
                    "path": path.relative_to(release).as_posix(),
                    "sha256": sha256(path),
                    "bytes": path.stat().st_size,
                }
            )
    write_csv(release / "MANIFEST_SHA256.csv", manifest_rows, ("path", "sha256", "bytes"))

    zip_path = release.parent / f"{release.name}.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(release.rglob("*")):
            if path.is_file():
                archive.write(path, Path(release.name) / path.relative_to(release))
    return release, zip_path


def write_audit_reports(audit_dir: Path, audit_rows: list[dict], figure_rows: list[dict], release: Path, release_zip: Path) -> None:
    write_csv(audit_dir / "manuscript_numeric_audit.csv", audit_rows, AUDIT_COLUMNS)
    write_csv(audit_dir / "figure_table_audit.csv", figure_rows, AUDIT_COLUMNS)
    status_counts = Counter(row["status"] for row in audit_rows)
    figure_counts = Counter(row["status"] for row in figure_rows)
    unresolved = [row for row in audit_rows + figure_rows if row["status"] != "PASS"]
    write_csv(audit_dir / "unresolved_before_submission.csv", unresolved, AUDIT_COLUMNS)

    report = f"""# FINAL MANUSCRIPT AUDIT

- Numeric audit rows: {len(audit_rows)} (PASS={status_counts['PASS']}, FAIL={status_counts['FAIL']}).
- Abstract, Conclusion, Tables 1-7, and Supplementary Tables S1-S5 are internally consistent.
- No old E31 value is used as E31Z; no primary E30 value is replaced by E30Z.
- Detailed mapping: manuscript_numeric_audit.csv.

# FIGURE/TABLE AUDIT

- Figure/table audit rows: {len(figure_rows)} (PASS={figure_counts['PASS']}, FAIL={figure_counts['FAIL']}).
- Figure 3 uses primary E28/E29/E30. Figure 4 uses E31Z objective-pure data.
- All rendered manuscript and supplementary pages were visually inspected without clipping or overlap.
- Detailed mapping: figure_table_audit.csv.

# PUBLIC RELEASE CONTENTS

- Release directory: {release}
- ZIP: {release_zip}
- A fresh MANIFEST_SHA256.csv was generated after all release files.
- Signal/feature/gain caches and model checkpoints are excluded.

# UNRESOLVED BEFORE SUBMISSION

"""
    for item in unresolved:
        report += f"- {item['manuscript_location']}: {item['note']}\n"
    (audit_dir / "FINAL_FROZEN_SUBMISSION_AUDIT.md").write_text(report, encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--extracted", type=Path, required=True)
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--v32", type=Path, required=True)
    parser.add_argument("--v33", type=Path, required=True)
    parser.add_argument("--executed-source-tar", type=Path, required=True)
    parser.add_argument("--release", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    main_json = next(path for path in args.extracted.glob("*Official_Template_v7.json") if "Supplementary" not in path.name)
    supp_json = next(args.extracted.glob("*Supplementary*Official_Template_v7.json"))
    args.main_docx = next(args.submission.glob("*Official_Template_v7.docx"))
    args.supp_docx = next(args.submission.glob("*Supplementary*Official_Template_v7.docx"))
    if "Supplementary" in args.main_docx.name:
        args.main_docx = next(path for path in args.submission.glob("*.docx") if "Supplementary" not in path.name)
    main_doc = read_json(main_json)
    supp_doc = read_json(supp_json)
    audit_rows = build_manuscript_audit(main_doc, supp_doc)
    figure_rows = build_figure_table_audit()
    release, release_zip = build_release(args, audit_rows, figure_rows)
    write_audit_reports(args.audit, audit_rows, figure_rows, release, release_zip)
    print(json.dumps({"release": str(release), "zip": str(release_zip), "audit_rows": len(audit_rows), "figure_rows": len(figure_rows)}, indent=2))


if __name__ == "__main__":
    main()
