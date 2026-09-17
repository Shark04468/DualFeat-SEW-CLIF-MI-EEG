"""DASP-SNN V6.2-R1 P0 protocol, provenance, and artifact contracts.

The helpers in this module deliberately do not import training frameworks.  P0
must be usable by every baseline and model runner before any EEG preprocessing
or checkpoint selection is allowed to begin.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, is_dataclass
import hashlib
import json
import math
from numbers import Integral, Real
from pathlib import Path
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from dpc_snn.utils.io import ensure_dir, read_json, save_npz, write_csv, write_json


FINGERPRINT_SCHEMA = "dasp-snn-v6.2-r1-p0-fingerprint/v1"
ARTIFACT_MANIFEST_SCHEMA = "dasp-snn-v6.2-r1-run-artifacts/v1"

REQUIRED_METADATA_FIELDS = (
    "dataset",
    "subject",
    "session",
    "run",
    "trial_id",
    "class",
    "sfreq",
    "ch_names",
    "epoch_tmin",
    "epoch_tmax",
)
FINGERPRINT_COMPONENTS = (
    "resolved_config",
    "source",
    "data",
    "split",
    "augmentation",
    "prior",
    "checkpoint",
    "environment",
)
PREDICTION_FIELDS = (
    "logits",
    "probabilities",
    "pred",
    "label",
    "subject",
    "session",
    "run",
    "trial_id",
    "seed",
    "model",
)
DEFAULT_REQUIRED_RUN_ARTIFACTS = (
    "resolved_config.yaml",
    "manifest.json",
    "source_fingerprint.json",
    "split_manifest.json",
    "augmentation_manifest.json",
    "history.csv",
    "predictions.npz",
    "predictions.csv",
    "metrics.json",
    "best.pt",
    "last.pt",
    "runtime_status.json",
    "stdout.log",
    "stderr.log",
)


class V62ProtocolError(ValueError):
    """Base class for a V6.2-R1 P0 protocol violation."""


class MetadataValidationError(V62ProtocolError):
    """Raised when trial metadata is absent, ambiguous, or inconsistent."""


class SessionLeakageError(V62ProtocolError):
    """Raised when Session E reaches a fitting or selection operation."""


class ResumeFingerprintMismatch(V62ProtocolError):
    """Raised when a run is resumed under a different scientific contract."""


class PredictionSchemaError(V62ProtocolError):
    """Raised when per-trial predictions do not satisfy the shared schema."""


class ArtifactManifestError(V62ProtocolError):
    """Raised when a formal run is missing or has modified artifacts."""


def _python_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def _is_non_string_sequence(value: Any) -> bool:
    if isinstance(value, np.ndarray):
        return value.ndim > 0
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def _column_value(value: Any, index: int, count: int, field: str) -> Any:
    """Read one value from a scalar-or-column metadata mapping."""

    value = _python_scalar(value)
    if field == "ch_names":
        if isinstance(value, np.ndarray) and value.ndim >= 2:
            if value.shape[0] != count:
                raise MetadataValidationError(
                    f"metadata field 'ch_names' has {value.shape[0]} rows; expected {count}"
                )
            return value[index]
        if _is_non_string_sequence(value):
            values = list(value)
            if values and _is_non_string_sequence(values[0]):
                if len(values) != count:
                    raise MetadataValidationError(
                        f"metadata field 'ch_names' has {len(values)} rows; expected {count}"
                    )
                return values[index]
        return value

    if not _is_non_string_sequence(value):
        return value
    values = list(value)
    if len(values) != count:
        raise MetadataValidationError(
            f"metadata field {field!r} has {len(values)} values; expected {count}"
        )
    return values[index]


def _metadata_rows(metadata: Any) -> list[dict[str, Any]]:
    if isinstance(metadata, Mapping):
        missing = [field for field in REQUIRED_METADATA_FIELDS if field not in metadata]
        if missing:
            raise MetadataValidationError(f"trial metadata missing required fields: {missing}")
        trial_ids = metadata["trial_id"]
        count = len(trial_ids) if _is_non_string_sequence(trial_ids) else 1
        if count == 0:
            raise MetadataValidationError("trial metadata must contain at least one trial")
        return [
            {
                field: _column_value(metadata[field], index, count, field)
                for field in REQUIRED_METADATA_FIELDS
            }
            for index in range(count)
        ]

    if not _is_non_string_sequence(metadata):
        raise MetadataValidationError("trial metadata must be a row sequence or column mapping")
    rows = list(metadata)
    if not rows:
        raise MetadataValidationError("trial metadata must contain at least one trial")
    if any(not isinstance(row, Mapping) for row in rows):
        raise MetadataValidationError("every trial metadata row must be a mapping")
    return [dict(row) for row in rows]


def _required_identifier(value: Any, field: str, row_index: int) -> str:
    value = _python_scalar(value)
    if isinstance(value, bool) or not isinstance(value, (str, Integral)):
        raise MetadataValidationError(
            f"metadata row {row_index} field {field!r} must be a non-empty string or integer"
        )
    normalized = str(value).strip()
    if not normalized:
        raise MetadataValidationError(f"metadata row {row_index} field {field!r} must not be empty")
    return normalized


def _finite_float(value: Any, field: str, row_index: int) -> float:
    value = _python_scalar(value)
    if isinstance(value, bool) or not isinstance(value, Real):
        raise MetadataValidationError(
            f"metadata row {row_index} field {field!r} must be a finite number"
        )
    result = float(value)
    if not math.isfinite(result):
        raise MetadataValidationError(f"metadata row {row_index} field {field!r} must be finite")
    return result


def _channel_names(value: Any, row_index: int) -> list[str]:
    if not _is_non_string_sequence(value):
        raise MetadataValidationError(
            f"metadata row {row_index} field 'ch_names' must be a non-empty channel sequence"
        )
    names = [_python_scalar(item) for item in list(value)]
    if not names or any(not isinstance(name, str) or not name.strip() for name in names):
        raise MetadataValidationError(
            f"metadata row {row_index} field 'ch_names' contains an invalid channel name"
        )
    normalized = [name.strip() for name in names]
    if len(set(normalized)) != len(normalized):
        raise MetadataValidationError(
            f"metadata row {row_index} field 'ch_names' contains duplicate channels"
        )
    return normalized


def _class_label(value: Any, row_index: int) -> int | str:
    value = _python_scalar(value)
    if isinstance(value, bool) or not isinstance(value, (Integral, str)):
        raise MetadataValidationError(
            f"metadata row {row_index} field 'class' must be an integer or non-empty string"
        )
    if isinstance(value, str):
        value = value.strip()
        if not value:
            raise MetadataValidationError(
                f"metadata row {row_index} field 'class' must not be empty"
            )
        return value
    return int(value)


def trial_identity(row: Mapping[str, Any]) -> tuple[str, str, str, str, str]:
    """Return the globally unambiguous identity of one physical trial."""

    return tuple(str(row[field]) for field in ("dataset", "subject", "session", "run", "trial_id"))  # type: ignore[return-value]


def validate_trial_metadata(
    metadata: Any,
    *,
    allowed_sessions: Sequence[str] = ("T", "E"),
) -> list[dict[str, Any]]:
    """Validate and normalize row-wise or column-wise trial metadata.

    Identifiers are normalized to strings, numerical acquisition fields to
    floats, and channel names to ordered lists.  Acquisition metadata must be
    constant within each dataset/subject/session and physical trial identities
    must be unique.
    """

    rows = _metadata_rows(metadata)
    sessions = {str(session) for session in allowed_sessions}
    normalized: list[dict[str, Any]] = []
    seen: dict[tuple[str, str, str, str, str], int] = {}
    acquisition: dict[tuple[str, str, str], tuple[float, tuple[str, ...], float, float]] = {}

    for index, row in enumerate(rows):
        missing = [field for field in REQUIRED_METADATA_FIELDS if field not in row]
        if missing:
            raise MetadataValidationError(
                f"metadata row {index} missing required fields: {missing}"
            )
        dataset = row["dataset"]
        if not isinstance(_python_scalar(dataset), str):
            raise MetadataValidationError(
                f"metadata row {index} field 'dataset' must be a non-empty string"
            )
        dataset_id = _required_identifier(dataset, "dataset", index)
        subject = _required_identifier(row["subject"], "subject", index)
        session = _required_identifier(row["session"], "session", index)
        run = _required_identifier(row["run"], "run", index)
        trial_id = _required_identifier(row["trial_id"], "trial_id", index)
        if session not in sessions:
            raise MetadataValidationError(
                f"metadata row {index} session {session!r} is not one of {sorted(sessions)}"
            )

        sfreq = _finite_float(row["sfreq"], "sfreq", index)
        if sfreq <= 0.0:
            raise MetadataValidationError(f"metadata row {index} field 'sfreq' must be > 0")
        epoch_tmin = _finite_float(row["epoch_tmin"], "epoch_tmin", index)
        epoch_tmax = _finite_float(row["epoch_tmax"], "epoch_tmax", index)
        if epoch_tmin >= epoch_tmax:
            raise MetadataValidationError(f"metadata row {index} requires epoch_tmin < epoch_tmax")
        if (epoch_tmax - epoch_tmin) * sfreq < 1.0:
            raise MetadataValidationError(
                f"metadata row {index} epoch bounds contain fewer than one sample"
            )
        ch_names = _channel_names(row["ch_names"], index)
        item = {
            "dataset": dataset_id,
            "subject": subject,
            "session": session,
            "run": run,
            "trial_id": trial_id,
            "class": _class_label(row["class"], index),
            "sfreq": sfreq,
            "ch_names": ch_names,
            "epoch_tmin": epoch_tmin,
            "epoch_tmax": epoch_tmax,
        }
        identity = trial_identity(item)
        if identity in seen:
            raise MetadataValidationError(
                f"duplicate physical trial identity at rows {seen[identity]} and {index}: {identity}"
            )
        seen[identity] = index

        session_key = (dataset_id, subject, session)
        acquisition_value = (sfreq, tuple(ch_names), epoch_tmin, epoch_tmax)
        if session_key in acquisition and acquisition[session_key] != acquisition_value:
            raise MetadataValidationError(
                "inconsistent sfreq/ch_names/epoch bounds within "
                f"dataset={dataset_id}, subject={subject}, session={session}"
            )
        acquisition[session_key] = acquisition_value
        normalized.append(item)

    return normalized


def assert_session_t_only(metadata: Any, *, purpose: str = "fitting") -> bool:
    """Reject any non-T trial in an operation that may affect model selection."""

    rows = validate_trial_metadata(metadata)
    leaked = [trial_identity(row) for row in rows if row["session"] != "T"]
    if leaked:
        raise SessionLeakageError(
            f"Session E leakage into {purpose}: {len(leaked)} non-T trial(s), first={leaked[0]}"
        )
    return True


def _validated_optional(metadata: Any | None) -> list[dict[str, Any]]:
    if metadata is None:
        return []
    if _is_non_string_sequence(metadata) and len(metadata) == 0:
        return []
    return validate_trial_metadata(metadata)


def _identity_set(rows: Sequence[Mapping[str, Any]]) -> set[tuple[str, str, str, str, str]]:
    return {trial_identity(row) for row in rows}


def assert_t_e_isolation(
    training_metadata: Any,
    evaluation_metadata: Any | None = None,
    *,
    validation_metadata: Any | None = None,
    augmentation_parent_metadata: Any | None = None,
    normalization_metadata: Any | None = None,
    ea_metadata: Any | None = None,
    prior_metadata: Any | None = None,
    checkpoint_metadata: Any | None = None,
    hpo_metadata: Any | None = None,
) -> bool:
    """Assert the complete Session-T fitting / Session-E evaluation boundary.

    Augmentation parents and fold-local fit statistics must be subsets of the
    inner-training trials.  Validation, checkpoint selection, and HPO may read
    Session T only.  Evaluation rows, when supplied, must be Session E only.
    """

    training = validate_trial_metadata(training_metadata)
    validation = _validated_optional(validation_metadata)
    evaluation = _validated_optional(evaluation_metadata)
    auxiliary = {
        "augmentation": _validated_optional(augmentation_parent_metadata),
        "normalization": _validated_optional(normalization_metadata),
        "euclidean_alignment": _validated_optional(ea_metadata),
        "prior": _validated_optional(prior_metadata),
        "checkpoint_selection": _validated_optional(checkpoint_metadata),
        "hpo": _validated_optional(hpo_metadata),
    }

    for purpose, rows in (("training", training), ("validation", validation), *auxiliary.items()):
        leaked = [trial_identity(row) for row in rows if row["session"] != "T"]
        if leaked:
            raise SessionLeakageError(
                f"Session E leakage into {purpose}: first leaked trial={leaked[0]}"
            )
    non_evaluation = [trial_identity(row) for row in evaluation if row["session"] != "E"]
    if non_evaluation:
        raise SessionLeakageError(
            "held-out evaluation must contain Session E only; "
            f"first invalid trial={non_evaluation[0]}"
        )

    train_ids = _identity_set(training)
    validation_ids = _identity_set(validation)
    evaluation_ids = _identity_set(evaluation)
    if train_ids & validation_ids:
        raise SessionLeakageError("training and validation contain overlapping physical trials")
    if (train_ids | validation_ids) & evaluation_ids:
        raise SessionLeakageError("Session-T fitting/selection overlaps held-out evaluation trials")

    for purpose in ("augmentation", "normalization", "euclidean_alignment", "prior"):
        outside = _identity_set(auxiliary[purpose]) - train_ids
        if outside:
            raise SessionLeakageError(
                f"{purpose} provenance is not a subset of the current inner-training fold: "
                f"first={sorted(outside)[0]}"
            )
    selectable = train_ids | validation_ids
    for purpose in ("checkpoint_selection", "hpo"):
        outside = _identity_set(auxiliary[purpose]) - selectable
        if outside:
            raise SessionLeakageError(
                f"{purpose} provenance is outside Session-T train/validation: "
                f"first={sorted(outside)[0]}"
            )
    return True


def _checked_indices(indices: Any, count: int, name: str) -> np.ndarray:
    array = np.asarray(indices)
    if array.ndim != 1 or array.dtype.kind not in "iu":
        raise SessionLeakageError(f"{name} indices must be a one-dimensional integer array")
    array = array.astype(np.int64, copy=False)
    if array.size != np.unique(array).size:
        raise SessionLeakageError(f"{name} indices contain duplicates")
    if array.size and (int(array.min()) < 0 or int(array.max()) >= count):
        raise SessionLeakageError(f"{name} indices are outside metadata bounds [0, {count})")
    return array


def assert_indexed_t_e_isolation(
    metadata: Any,
    train_indices: Any,
    evaluation_indices: Any,
    *,
    validation_indices: Any = (),
) -> bool:
    """Index-based adapter for runners that retain one complete metadata table."""

    rows = validate_trial_metadata(metadata)
    train = _checked_indices(train_indices, len(rows), "training")
    validation = _checked_indices(validation_indices, len(rows), "validation")
    evaluation = _checked_indices(evaluation_indices, len(rows), "evaluation")
    if (
        set(train) & set(validation)
        or set(train) & set(evaluation)
        or set(validation) & set(evaluation)
    ):
        raise SessionLeakageError("training, validation, and evaluation indices must be disjoint")
    return assert_t_e_isolation(
        [rows[index] for index in train],
        [rows[index] for index in evaluation],
        validation_metadata=[rows[index] for index in validation],
    )


def session_t_run_grouped_folds(
    metadata: Any,
    *,
    n_splits: int | None = None,
    seed: int = 0,
    shuffle: bool = True,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Build deterministic Session-T folds while keeping each run intact."""

    rows = validate_trial_metadata(metadata)
    assert_session_t_only(rows, purpose="run-grouped fold construction")
    groups: dict[tuple[str, str, str], list[int]] = {}
    for index, row in enumerate(rows):
        group = (row["dataset"], row["subject"], row["run"])
        groups.setdefault(group, []).append(index)
    ordered_groups = sorted(groups, key=canonical_json)
    if len(ordered_groups) < 2:
        raise V62ProtocolError("run-grouped cross-validation requires at least two runs")
    if n_splits is None:
        n_splits = len(ordered_groups)
    if isinstance(n_splits, bool) or not isinstance(n_splits, Integral):
        raise V62ProtocolError("n_splits must be an integer")
    n_splits = int(n_splits)
    if n_splits < 2 or n_splits > len(ordered_groups):
        raise V62ProtocolError(f"n_splits must be in [2, {len(ordered_groups)}], got {n_splits}")
    if shuffle:
        rng = np.random.default_rng(int(seed))
        order = rng.permutation(len(ordered_groups))
        ordered_groups = [ordered_groups[int(index)] for index in order]

    group_chunks = [ordered_groups[offset::n_splits] for offset in range(n_splits)]
    all_indices = np.arange(len(rows), dtype=np.int64)
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for chunk in group_chunks:
        heldout_groups = set(chunk)
        validation = np.asarray(
            [index for group in chunk for index in groups[group]],
            dtype=np.int64,
        )
        validation.sort()
        training = np.setdiff1d(all_indices, validation, assume_unique=True)
        if training.size == 0 or validation.size == 0:
            raise V62ProtocolError("run-grouped cross-validation produced an empty fold")
        train_groups = {
            (rows[int(index)]["dataset"], rows[int(index)]["subject"], rows[int(index)]["run"])
            for index in training
        }
        if train_groups & heldout_groups:
            raise V62ProtocolError("internal error: a physical run was split across a fold")
        folds.append((training, validation))
    return folds


def _canonical_value(value: Any) -> Any:
    value = _python_scalar(value)
    if value is None or isinstance(value, (bool, str, Integral)):
        return int(value) if isinstance(value, Integral) and not isinstance(value, bool) else value
    if isinstance(value, Real):
        number = float(value)
        if not math.isfinite(number):
            raise V62ProtocolError("canonical JSON does not permit NaN or infinity")
        return 0.0 if number == 0.0 else number
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, (bytes, bytearray, memoryview)):
        raw = bytes(value)
        return {"__bytes__": {"size": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}}
    if isinstance(value, np.ndarray):
        array = np.asarray(value)
        if array.dtype.hasobject:
            return {"__ndarray_values__": _canonical_value(array.tolist())}
        if array.dtype.kind in "fc" and not np.isfinite(array).all():
            raise V62ProtocolError("canonical ndarray contains NaN or infinity")
        contiguous = np.ascontiguousarray(array)
        return {
            "__ndarray__": {
                "dtype": contiguous.dtype.str,
                "shape": list(contiguous.shape),
                "sha256": hashlib.sha256(contiguous.tobytes(order="C")).hexdigest(),
            }
        }
    if is_dataclass(value) and not isinstance(value, type):
        return _canonical_value(asdict(value))
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise V62ProtocolError("canonical JSON mapping keys must be strings")
            result[key] = _canonical_value(item)
        return result
    if isinstance(value, (set, frozenset)):
        items = [_canonical_value(item) for item in value]
        return sorted(
            items, key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":"))
        )
    if _is_non_string_sequence(value):
        return [_canonical_value(item) for item in value]
    raise V62ProtocolError(f"unsupported canonical JSON type: {type(value).__name__}")


def canonical_json(value: Any) -> str:
    """Serialize supported scientific state to deterministic canonical JSON."""

    return json.dumps(
        _canonical_value(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def sha256_fingerprint(value: Any) -> str:
    """Return the SHA-256 digest of canonical UTF-8 JSON."""

    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Hash one file without loading it into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def build_run_fingerprint(
    *,
    resolved_config: Any,
    source: Any,
    data: Any,
    split: Any,
    augmentation: Any,
    prior: Any,
    checkpoint: Any,
    environment: Any,
) -> dict[str, Any]:
    """Build all eight mandatory component hashes and one combined run hash."""

    payloads = {
        "resolved_config": resolved_config,
        "source": source,
        "data": data,
        "split": split,
        "augmentation": augmentation,
        "prior": prior,
        "checkpoint": checkpoint,
        "environment": environment,
    }
    components = {name: sha256_fingerprint(payloads[name]) for name in FINGERPRINT_COMPONENTS}
    header = {
        "schema": FINGERPRINT_SCHEMA,
        "algorithm": "sha256",
        "components": components,
    }
    return {**header, "combined_sha256": sha256_fingerprint(header)}


def validate_fingerprint_manifest(fingerprint: Mapping[str, Any]) -> dict[str, Any]:
    """Validate component completeness and the combined digest."""

    expected_keys = {"schema", "algorithm", "components", "combined_sha256"}
    if set(fingerprint) != expected_keys:
        raise V62ProtocolError(
            f"fingerprint fields must be exactly {sorted(expected_keys)}, got {sorted(fingerprint)}"
        )
    if fingerprint["schema"] != FINGERPRINT_SCHEMA or fingerprint["algorithm"] != "sha256":
        raise V62ProtocolError("unsupported fingerprint schema or algorithm")
    components = fingerprint["components"]
    if not isinstance(components, Mapping) or set(components) != set(FINGERPRINT_COMPONENTS):
        raise V62ProtocolError(
            f"fingerprint components must be exactly {list(FINGERPRINT_COMPONENTS)}"
        )
    for name, digest in components.items():
        if not isinstance(digest, str) or len(digest) != 64:
            raise V62ProtocolError(f"fingerprint component {name!r} is not a SHA-256 digest")
        try:
            int(digest, 16)
        except ValueError as exc:
            raise V62ProtocolError(f"fingerprint component {name!r} is not hexadecimal") from exc
    header = {
        "schema": fingerprint["schema"],
        "algorithm": fingerprint["algorithm"],
        "components": dict(components),
    }
    expected = sha256_fingerprint(header)
    if fingerprint["combined_sha256"] != expected:
        raise V62ProtocolError("combined fingerprint does not match its component hashes")
    return {**header, "combined_sha256": expected}


def _load_fingerprint(value: Mapping[str, Any] | str | Path) -> dict[str, Any]:
    payload = read_json(value) if isinstance(value, (str, Path)) else dict(value)
    return validate_fingerprint_manifest(payload)


def validate_resume_fingerprint(
    saved: Mapping[str, Any] | str | Path,
    current: Mapping[str, Any] | str | Path,
) -> bool:
    """Permit resume only when the complete fingerprint manifests are equal."""

    saved_payload = _load_fingerprint(saved)
    current_payload = _load_fingerprint(current)
    if canonical_json(saved_payload) == canonical_json(current_payload):
        return True
    changed = [
        name
        for name in FINGERPRINT_COMPONENTS
        if saved_payload["components"][name] != current_payload["components"][name]
    ]
    if saved_payload["combined_sha256"] != current_payload["combined_sha256"] and not changed:
        changed.append("combined_sha256")
    raise ResumeFingerprintMismatch(
        "resume fingerprint mismatch; changed components: " + ", ".join(changed)
    )


def write_fingerprint_manifest(path: str | Path, fingerprint: Mapping[str, Any]) -> Path:
    """Validate then atomically write a fingerprint manifest."""

    return write_json(path, validate_fingerprint_manifest(fingerprint))


def _broadcast_vector(value: Any, count: int, field: str) -> np.ndarray:
    value = _python_scalar(value)
    if isinstance(value, (str, bytes)) or not _is_non_string_sequence(value):
        return np.asarray([value] * count)
    array = np.asarray(value)
    if array.ndim != 1 or array.shape[0] != count:
        raise PredictionSchemaError(
            f"prediction field {field!r} must be scalar or shape ({count},), got {array.shape}"
        )
    return array


def _integer_prediction_vector(value: Any, count: int, field: str) -> np.ndarray:
    array = _broadcast_vector(value, count, field)
    if array.dtype.kind not in "iu" or array.dtype.kind == "b":
        raise PredictionSchemaError(f"prediction field {field!r} must contain integers")
    return array.astype(np.int64, copy=False)


def _prediction_arrays(
    *,
    logits: Any,
    probabilities: Any,
    pred: Any,
    label: Any,
    subject: Any,
    session: Any,
    run: Any,
    trial_id: Any,
    seed: Any,
    model: Any,
) -> dict[str, np.ndarray]:
    logits_array = np.asarray(logits)
    probabilities_array = np.asarray(probabilities)
    if logits_array.ndim != 2 or logits_array.shape[0] == 0 or logits_array.shape[1] == 0:
        raise PredictionSchemaError("logits must have non-empty shape (trials, classes)")
    if probabilities_array.shape != logits_array.shape:
        raise PredictionSchemaError("probabilities must have the same shape as logits")
    if logits_array.dtype.kind not in "fiu" or probabilities_array.dtype.kind not in "fiu":
        raise PredictionSchemaError("logits and probabilities must be numeric")
    logits_array = logits_array.astype(np.float32, copy=False)
    probabilities_array = probabilities_array.astype(np.float32, copy=False)
    if not np.isfinite(logits_array).all() or not np.isfinite(probabilities_array).all():
        raise PredictionSchemaError("logits and probabilities must be finite")
    if np.any(probabilities_array < 0.0) or np.any(probabilities_array > 1.0):
        raise PredictionSchemaError("probabilities must lie in [0, 1]")
    if not np.allclose(probabilities_array.sum(axis=1), 1.0, atol=1e-5, rtol=1e-5):
        raise PredictionSchemaError("each probability row must sum to one")

    count, classes = logits_array.shape
    pred_array = _integer_prediction_vector(pred, count, "pred")
    label_array = _integer_prediction_vector(label, count, "label")
    if np.any(pred_array < 0) or np.any(pred_array >= classes):
        raise PredictionSchemaError("pred contains an out-of-range class index")
    if not np.array_equal(pred_array, probabilities_array.argmax(axis=1)):
        raise PredictionSchemaError("pred must equal argmax(probabilities) for every trial")
    subject_array = _broadcast_vector(subject, count, "subject").astype(np.str_)
    session_array = _broadcast_vector(session, count, "session").astype(np.str_)
    run_array = _broadcast_vector(run, count, "run").astype(np.str_)
    trial_id_array = _broadcast_vector(trial_id, count, "trial_id").astype(np.str_)
    seed_array = _integer_prediction_vector(seed, count, "seed")
    model_array = _broadcast_vector(model, count, "model").astype(np.str_)
    for field, array in (
        ("subject", subject_array),
        ("session", session_array),
        ("run", run_array),
        ("trial_id", trial_id_array),
        ("model", model_array),
    ):
        if any(not str(item).strip() for item in array):
            raise PredictionSchemaError(f"prediction field {field!r} contains an empty value")
    if not set(session_array.tolist()).issubset({"T", "E", "S1", "S2"}):
        raise PredictionSchemaError(
            "prediction session values must be 'T', 'E', 'S1' or 'S2'"
        )
    identities = list(
        zip(
            subject_array.tolist(),
            session_array.tolist(),
            run_array.tolist(),
            trial_id_array.tolist(),
            seed_array.tolist(),
            model_array.tolist(),
            strict=True,
        )
    )
    if len(set(identities)) != count:
        raise PredictionSchemaError("prediction rows contain duplicate trial/seed/model identities")
    return {
        "logits": logits_array,
        "probabilities": probabilities_array,
        "pred": pred_array,
        "label": label_array,
        "subject": subject_array,
        "session": session_array,
        "run": run_array,
        "trial_id": trial_id_array,
        "seed": seed_array,
        "model": model_array,
    }


def write_trial_predictions(
    output_dir: str | Path,
    *,
    logits: Any,
    probabilities: Any,
    pred: Any,
    label: Any,
    subject: Any,
    session: Any,
    run: Any,
    trial_id: Any,
    seed: Any,
    model: Any,
    basename: str = "predictions",
) -> dict[str, Path]:
    """Validate and atomically write the shared per-trial NPZ and CSV files."""

    if not basename or Path(basename).name != basename:
        raise PredictionSchemaError("prediction basename must be one plain filename stem")
    arrays = _prediction_arrays(
        logits=logits,
        probabilities=probabilities,
        pred=pred,
        label=label,
        subject=subject,
        session=session,
        run=run,
        trial_id=trial_id,
        seed=seed,
        model=model,
    )
    directory = ensure_dir(output_dir)
    npz_path = save_npz(directory / f"{basename}.npz", **arrays)
    rows = [
        {
            "logits": canonical_json(arrays["logits"][index].tolist()),
            "probabilities": canonical_json(arrays["probabilities"][index].tolist()),
            "pred": int(arrays["pred"][index]),
            "label": int(arrays["label"][index]),
            "subject": str(arrays["subject"][index]),
            "session": str(arrays["session"][index]),
            "run": str(arrays["run"][index]),
            "trial_id": str(arrays["trial_id"][index]),
            "seed": int(arrays["seed"][index]),
            "model": str(arrays["model"][index]),
        }
        for index in range(arrays["pred"].shape[0])
    ]
    csv_path = write_csv(directory / f"{basename}.csv", rows, fieldnames=list(PREDICTION_FIELDS))
    return {"npz": npz_path, "csv": csv_path}


def validate_prediction_schema(
    npz_path: str | Path,
    csv_path: str | Path | None = None,
) -> dict[str, int]:
    """Validate prediction archive keys, dimensions, and optional CSV parity."""

    with np.load(npz_path, allow_pickle=False) as archive:
        if set(archive.files) != set(PREDICTION_FIELDS):
            raise PredictionSchemaError(
                f"prediction NPZ fields must be exactly {list(PREDICTION_FIELDS)}"
            )
        arrays = {field: archive[field] for field in PREDICTION_FIELDS}
    checked = _prediction_arrays(**arrays)
    count, classes = checked["logits"].shape
    if csv_path is not None:
        with Path(csv_path).open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != list(PREDICTION_FIELDS):
                raise PredictionSchemaError(
                    f"prediction CSV fields must be exactly {list(PREDICTION_FIELDS)}"
                )
            rows = list(reader)
        if len(rows) != count:
            raise PredictionSchemaError("prediction CSV and NPZ have different trial counts")
        scalar_fields = ("pred", "label", "subject", "session", "run", "trial_id", "seed", "model")
        for index, row in enumerate(rows):
            for field in scalar_fields:
                if row[field] != str(checked[field][index]):
                    raise PredictionSchemaError(
                        f"prediction CSV/NPZ mismatch at row {index}, field {field!r}"
                    )
            for field in ("logits", "probabilities"):
                try:
                    csv_values = np.asarray(json.loads(row[field]), dtype=np.float32)
                except (json.JSONDecodeError, TypeError, ValueError) as exc:
                    raise PredictionSchemaError(
                        f"prediction CSV row {index} field {field!r} is not a numeric JSON array"
                    ) from exc
                if not np.array_equal(csv_values, checked[field][index]):
                    raise PredictionSchemaError(
                        f"prediction CSV/NPZ mismatch at row {index}, field {field!r}"
                    )
    return {"n_trials": int(count), "n_classes": int(classes)}


def _normalize_required_files(required_files: Sequence[str]) -> tuple[str, ...]:
    normalized: list[str] = []
    for item in required_files:
        if not isinstance(item, str) or not item.strip():
            raise ArtifactManifestError("required artifact names must be non-empty strings")
        path = Path(item)
        if path.is_absolute() or ".." in path.parts:
            raise ArtifactManifestError(
                f"artifact path must stay inside the run directory: {item!r}"
            )
        name = path.as_posix()
        if name in normalized:
            raise ArtifactManifestError(f"duplicate required artifact: {name}")
        normalized.append(name)
    if "manifest.json" not in normalized:
        raise ArtifactManifestError("required artifact set must include manifest.json")
    return tuple(normalized)


def _artifact_path(run_dir: Path, name: str) -> Path:
    return run_dir.joinpath(*Path(name).parts)


def check_required_run_artifacts(
    run_dir: str | Path,
    *,
    required_files: Sequence[str] = DEFAULT_REQUIRED_RUN_ARTIFACTS,
) -> dict[str, Path]:
    """Require every declared run artifact to exist as a regular file."""

    directory = Path(run_dir)
    required = _normalize_required_files(required_files)
    paths = {name: _artifact_path(directory, name) for name in required}
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise ArtifactManifestError(f"run is missing required artifacts: {missing}")
    return paths


def write_run_artifact_manifest(
    run_dir: str | Path,
    *,
    required_files: Sequence[str] = DEFAULT_REQUIRED_RUN_ARTIFACTS,
    status: str = "completed",
) -> Path:
    """Hash required artifacts and atomically publish ``manifest.json`` last."""

    directory = ensure_dir(run_dir)
    required = _normalize_required_files(required_files)
    missing = [
        name
        for name in required
        if name != "manifest.json" and not _artifact_path(directory, name).is_file()
    ]
    if missing:
        raise ArtifactManifestError(f"cannot write manifest; missing required artifacts: {missing}")
    artifacts = []
    for name in required:
        if name == "manifest.json":
            continue
        path = _artifact_path(directory, name)
        artifacts.append(
            {
                "path": name,
                "size_bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    payload = {
        "schema": ARTIFACT_MANIFEST_SCHEMA,
        "status": str(status),
        "required_files": list(required),
        "artifacts": artifacts,
    }
    return write_json(directory / "manifest.json", payload)


def validate_run_artifact_manifest(
    run_dir: str | Path,
    *,
    required_files: Sequence[str] = DEFAULT_REQUIRED_RUN_ARTIFACTS,
    verify_hashes: bool = True,
    verify_prediction_schema: bool = False,
) -> dict[str, Any]:
    """Check required files and, by default, every recorded size and SHA-256."""

    directory = Path(run_dir)
    required = _normalize_required_files(required_files)
    paths = check_required_run_artifacts(directory, required_files=required)
    try:
        manifest = read_json(paths["manifest.json"])
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactManifestError("manifest.json is unreadable or invalid JSON") from exc
    expected_keys = {"schema", "status", "required_files", "artifacts"}
    if set(manifest) != expected_keys or manifest.get("schema") != ARTIFACT_MANIFEST_SCHEMA:
        raise ArtifactManifestError("manifest.json has an invalid V6.2 artifact schema")
    try:
        declared_required = _normalize_required_files(manifest["required_files"])
    except (TypeError, ArtifactManifestError) as exc:
        raise ArtifactManifestError("manifest.json has invalid required_files") from exc
    if declared_required != required:
        raise ArtifactManifestError("manifest required_files do not match the active protocol")
    if not isinstance(manifest["artifacts"], list):
        raise ArtifactManifestError("manifest artifacts must be a list")
    entries: dict[str, Mapping[str, Any]] = {}
    for entry in manifest["artifacts"]:
        if not isinstance(entry, Mapping) or set(entry) != {"path", "size_bytes", "sha256"}:
            raise ArtifactManifestError("manifest contains an invalid artifact entry")
        name = entry["path"]
        if not isinstance(name, str) or name in entries:
            raise ArtifactManifestError("manifest contains a duplicate or invalid artifact path")
        entries[name] = entry
    expected_entries = set(required) - {"manifest.json"}
    if set(entries) != expected_entries:
        missing = sorted(expected_entries - set(entries))
        extra = sorted(set(entries) - expected_entries)
        raise ArtifactManifestError(
            f"manifest artifact entries differ from protocol; missing={missing}, extra={extra}"
        )
    if verify_hashes:
        for name, entry in entries.items():
            path = paths[name]
            if entry["size_bytes"] != path.stat().st_size or entry["sha256"] != file_sha256(path):
                raise ArtifactManifestError(f"artifact size/hash mismatch: {name}")
    if verify_prediction_schema and {"predictions.npz", "predictions.csv"}.issubset(paths):
        validate_prediction_schema(paths["predictions.npz"], paths["predictions.csv"])
    return manifest


# Readable aliases for runner call sites.
build_session_t_run_grouped_folds = session_t_run_grouped_folds
build_protocol_fingerprint = build_run_fingerprint
assert_resume_compatible = validate_resume_fingerprint
record_trial_predictions = write_trial_predictions
create_run_artifact_manifest = write_run_artifact_manifest
