"""V8 accuracy-first protocol and provenance contracts.

V8 deliberately starts a new scientific namespace after the V7 delay gate
failed.  The repository may be copied without ``.git`` metadata, so formal
resume decisions are based on a complete content manifest rather than a commit
identifier.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
from pathlib import Path
from typing import Any, Literal

from dpc_snn.experiments.v62_protocol import (
    canonical_json,
    file_sha256,
    sha256_fingerprint,
    validate_trial_metadata,
)
from dpc_snn.utils.io import read_json, write_json


V8_PROTOCOL_ID = "dpc-snn-v8-accuracy-first"
V8_FINGERPRINT_SCHEMA = "dpc-snn-v8-accuracy-first-fingerprint/v1"
V8_FREEZE_SCHEMA = "dpc-snn-v8-architecture-freeze/v1"
V8_EXTERNAL_UNLOCK_SCHEMA = "dpc-snn-v8-openbmi-unlock/v1"
V8_FINGERPRINT_COMPONENTS = (
    "protocol",
    "resolved_run_config",
    "source_tree",
    "data",
    "split",
    "augmentation",
    "prior",
    "checkpoint",
    "environment",
)
V8_SOURCE_ROOTS = ("src", "scripts", "configs", "tests")
V8_SOURCE_SUFFIXES = frozenset({".py", ".yaml", ".yml", ".toml", ".txt", ".sh", ".ps1"})
V8_ROOT_SOURCE_FILES = (
    "pyproject.toml",
    "requirements.txt",
    "environment.yml",
)
V8_EXCLUDED_SOURCE_PARTS = frozenset({"__pycache__", ".pytest_cache", ".ruff_cache"})

V8Stage = Literal["development", "bci2a_evaluation", "openbmi_confirmation"]
V8Role = Literal[
    "training",
    "validation",
    "augmentation",
    "normalization",
    "prior",
    "checkpoint_selection",
    "hpo",
    "evaluation",
]

_FIT_ROLES = frozenset(
    {
        "training",
        "validation",
        "augmentation",
        "normalization",
        "prior",
        "checkpoint_selection",
        "hpo",
    }
)
_BCI2A_ALIASES = frozenset({"bci2a", "bciciv2a", "bci_competition_iv_2a"})
_OPENBMI_ALIASES = frozenset(
    {"openbmi", "openbmi_lee2019_mi", "lee2019_mi", "lee2019"}
)


class V8ProtocolError(ValueError):
    """Base class for a V8 scientific-contract violation."""


class V8DataAccessError(V8ProtocolError):
    """Raised when a locked dataset/session reaches a forbidden stage."""


class V8ResumeFingerprintMismatch(V8ProtocolError):
    """Raised when any formal run component changed before resume."""


class V8FreezeManifestError(V8ProtocolError):
    """Raised when a held-out evaluation freeze is incomplete or altered."""


def collect_source_tree_manifest(root: str | Path) -> dict[str, str]:
    """Hash every executable/configuration source file in the V8 repository.

    This is intentionally broader than a hand-maintained source list.  Adding,
    removing, or modifying a runner, test, model, or configuration changes the
    combined run fingerprint even when the checkout has no Git metadata.
    """

    root_path = Path(root).resolve()
    if not root_path.is_dir():
        raise V8ProtocolError(f"source root does not exist: {root_path}")
    manifest: dict[str, str] = {}
    for directory in V8_SOURCE_ROOTS:
        base = root_path / directory
        if not base.is_dir():
            raise V8ProtocolError(f"required source directory is missing: {base}")
        for path in sorted(base.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in V8_SOURCE_SUFFIXES:
                continue
            relative_parts = path.relative_to(root_path).parts
            if any(
                part in V8_EXCLUDED_SOURCE_PARTS or part.endswith(".egg-info")
                for part in relative_parts
            ):
                continue
            relative = path.relative_to(root_path).as_posix()
            manifest[relative] = file_sha256(path)
    for name in V8_ROOT_SOURCE_FILES:
        path = root_path / name
        if not path.is_file():
            raise V8ProtocolError(f"required source file is missing: {path}")
        manifest[name] = file_sha256(path)
    if not manifest:
        raise V8ProtocolError("source tree manifest is empty")
    return dict(sorted(manifest.items()))


def source_tree_digest(manifest: Mapping[str, str]) -> str:
    """Return a canonical digest while rejecting malformed source manifests."""

    if not manifest:
        raise V8ProtocolError("source tree manifest is empty")
    normalized: dict[str, str] = {}
    for name, digest in manifest.items():
        if not isinstance(name, str) or not name or "\\" in name:
            raise V8ProtocolError("source manifest paths must be non-empty POSIX paths")
        if not isinstance(digest, str) or len(digest) != 64:
            raise V8ProtocolError(f"source digest for {name!r} is not SHA-256")
        try:
            int(digest, 16)
        except ValueError as exc:
            raise V8ProtocolError(f"source digest for {name!r} is not hexadecimal") from exc
        normalized[name] = digest.lower()
    return sha256_fingerprint(dict(sorted(normalized.items())))


def build_v8_run_fingerprint(
    *,
    resolved_run_config: Any,
    source_tree: Mapping[str, str],
    data: Any,
    split: Any,
    augmentation: Any,
    prior: Any,
    checkpoint: Any,
    environment: Any,
) -> dict[str, Any]:
    """Build a complete V8 fingerprint for scientific resume decisions."""

    source_tree_digest(source_tree)
    payloads = {
        "protocol": {"id": V8_PROTOCOL_ID, "schema": V8_FINGERPRINT_SCHEMA},
        "resolved_run_config": resolved_run_config,
        "source_tree": dict(source_tree),
        "data": data,
        "split": split,
        "augmentation": augmentation,
        "prior": prior,
        "checkpoint": checkpoint,
        "environment": environment,
    }
    components = {
        name: sha256_fingerprint(payloads[name]) for name in V8_FINGERPRINT_COMPONENTS
    }
    header = {
        "schema": V8_FINGERPRINT_SCHEMA,
        "algorithm": "sha256",
        "components": components,
    }
    return {**header, "combined_sha256": sha256_fingerprint(header)}


def validate_v8_fingerprint(fingerprint: Mapping[str, Any]) -> dict[str, Any]:
    """Validate V8 fingerprint completeness and its combined digest."""

    required = {"schema", "algorithm", "components", "combined_sha256"}
    if set(fingerprint) != required:
        raise V8ProtocolError(
            f"V8 fingerprint fields must be exactly {sorted(required)}"
        )
    if fingerprint["schema"] != V8_FINGERPRINT_SCHEMA:
        raise V8ProtocolError("unsupported V8 fingerprint schema")
    if fingerprint["algorithm"] != "sha256":
        raise V8ProtocolError("V8 fingerprints require SHA-256")
    components = fingerprint["components"]
    if not isinstance(components, Mapping) or set(components) != set(
        V8_FINGERPRINT_COMPONENTS
    ):
        raise V8ProtocolError(
            f"V8 fingerprint components must be exactly {list(V8_FINGERPRINT_COMPONENTS)}"
        )
    for name, digest in components.items():
        if not isinstance(digest, str) or len(digest) != 64:
            raise V8ProtocolError(f"V8 component {name!r} is not SHA-256")
        try:
            int(digest, 16)
        except ValueError as exc:
            raise V8ProtocolError(f"V8 component {name!r} is not hexadecimal") from exc
    header = {
        "schema": V8_FINGERPRINT_SCHEMA,
        "algorithm": "sha256",
        "components": dict(components),
    }
    combined = sha256_fingerprint(header)
    if fingerprint["combined_sha256"] != combined:
        raise V8ProtocolError("V8 combined fingerprint does not match its components")
    return {**header, "combined_sha256": combined}


def _load_v8_fingerprint(value: Mapping[str, Any] | str | Path) -> dict[str, Any]:
    payload = read_json(value) if isinstance(value, (str, Path)) else dict(value)
    return validate_v8_fingerprint(payload)


def validate_v8_resume_fingerprint(
    saved: Mapping[str, Any] | str | Path,
    current: Mapping[str, Any] | str | Path,
) -> bool:
    """Allow resume only when all nine V8 fingerprint components match."""

    saved_payload = _load_v8_fingerprint(saved)
    current_payload = _load_v8_fingerprint(current)
    if canonical_json(saved_payload) == canonical_json(current_payload):
        return True
    changed = [
        name
        for name in V8_FINGERPRINT_COMPONENTS
        if saved_payload["components"][name] != current_payload["components"][name]
    ]
    raise V8ResumeFingerprintMismatch(
        "V8 resume fingerprint mismatch; changed components: " + ", ".join(changed)
    )


def write_v8_fingerprint(path: str | Path, fingerprint: Mapping[str, Any]) -> Path:
    """Validate and atomically write one V8 fingerprint manifest."""

    return write_json(path, validate_v8_fingerprint(fingerprint))


def build_v8_freeze_manifest(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Seal the complete pre-E6 architecture and analysis contract.

    The digest excludes only itself.  Formal E6-E9 runners validate this
    manifest before loading any held-out signal or label.
    """

    body = dict(payload)
    body.pop("combined_sha256", None)
    body.setdefault("schema", V8_FREEZE_SCHEMA)
    body.setdefault("protocol", V8_PROTOCOL_ID)
    return validate_v8_freeze_manifest(
        {**body, "combined_sha256": sha256_fingerprint(body)}
    )


def validate_v8_freeze_manifest(
    manifest: Mapping[str, Any] | str | Path,
    *,
    expected_source_tree_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate a frozen pre-E6 contract and its held-out-data lock history."""

    value = read_json(manifest) if isinstance(manifest, (str, Path)) else dict(manifest)
    required = {
        "schema",
        "protocol",
        "freeze_scope",
        "created_at_utc",
        "source_tree_sha256",
        "architecture",
        "training",
        "preprocessing",
        "augmentation",
        "checkpoint_rule",
        "analysis_plan",
        "development_evidence",
        "baselines",
        "heldout_access",
        "artifact_hashes",
        "combined_sha256",
    }
    if set(value) != required:
        raise V8FreezeManifestError(
            f"V8 freeze fields must be exactly {sorted(required)}"
        )
    if value["schema"] != V8_FREEZE_SCHEMA or value["protocol"] != V8_PROTOCOL_ID:
        raise V8FreezeManifestError("unsupported V8 freeze schema or protocol")
    if value["freeze_scope"] != "pre_E6_architecture_and_analysis":
        raise V8FreezeManifestError("V8 freeze has the wrong scope")
    source_digest = str(value["source_tree_sha256"])
    if len(source_digest) != 64:
        raise V8FreezeManifestError("V8 freeze source-tree digest is not SHA-256")
    try:
        int(source_digest, 16)
    except ValueError as exc:
        raise V8FreezeManifestError(
            "V8 freeze source-tree digest is not hexadecimal"
        ) from exc
    if expected_source_tree_sha256 is not None and source_digest != str(
        expected_source_tree_sha256
    ):
        raise V8FreezeManifestError("active source tree differs from the frozen tree")

    architecture = value["architecture"]
    if not isinstance(architecture, Mapping) or not isinstance(
        architecture.get("model_config"), Mapping
    ):
        raise V8FreezeManifestError("V8 freeze does not contain a resolved model config")
    if architecture.get("primary_variant") not in {
        "ann_residual",
        "plif_plain",
        "clif_plain",
        "sew_clif",
    }:
        raise V8FreezeManifestError("V8 freeze primary decoder variant is unknown")
    delay = architecture.get("delay")
    if not isinstance(delay, Mapping) or bool(delay.get("enabled")) != (
        delay.get("mode") != "off"
    ):
        raise V8FreezeManifestError("V8 freeze delay state is internally inconsistent")

    checkpoint = value["checkpoint_rule"]
    if (
        not isinstance(checkpoint, Mapping)
        or checkpoint.get("selection_data") != "BCI2a Session T development only"
        or int(checkpoint.get("final_epoch", 0)) < 1
        or int(checkpoint.get("scheduler_horizon", 0))
        < int(checkpoint.get("final_epoch", 0))
        or checkpoint.get("heldout_checkpoint_selection") is not False
    ):
        raise V8FreezeManifestError("V8 freeze checkpoint rule is invalid")

    heldout = value["heldout_access"]
    if (
        not isinstance(heldout, Mapping)
        or heldout.get("bci2a_session_e_accessed_before_freeze") is not False
        or heldout.get("openbmi_session_s2_accessed_before_freeze") is not False
        or heldout.get("status") != "sealed_until_validated_freeze"
    ):
        raise V8FreezeManifestError("held-out data were accessed before V8 freeze")
    analysis = value["analysis_plan"]
    if (
        not isinstance(analysis, Mapping)
        or analysis.get("primary_metric") != "subject_macro_accuracy"
        or analysis.get("subject_is_inferential_unit") is not True
        or analysis.get("post_E_tuning_allowed") is not False
    ):
        raise V8FreezeManifestError("V8 frozen analysis plan is incomplete")
    artifact_hashes = value["artifact_hashes"]
    if not isinstance(artifact_hashes, Mapping) or not artifact_hashes:
        raise V8FreezeManifestError("V8 freeze has no development artifact hashes")
    for name, digest in artifact_hashes.items():
        if not isinstance(name, str) or not isinstance(digest, str) or len(digest) != 64:
            raise V8FreezeManifestError(f"invalid frozen artifact digest for {name!r}")
        try:
            int(digest, 16)
        except ValueError as exc:
            raise V8FreezeManifestError(
                f"frozen artifact digest for {name!r} is not hexadecimal"
            ) from exc

    expected = sha256_fingerprint(
        {key: item for key, item in value.items() if key != "combined_sha256"}
    )
    if value["combined_sha256"] != expected:
        raise V8FreezeManifestError("V8 freeze digest does not match its contents")
    return value


def write_v8_freeze_manifest(path: str | Path, payload: Mapping[str, Any]) -> Path:
    """Build, validate and atomically publish one pre-E6 freeze manifest."""

    return write_json(path, build_v8_freeze_manifest(payload))


def build_v8_external_unlock_manifest(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Seal the external-confirmation protocol before OpenBMI Session S2."""

    body = dict(payload)
    body.pop("combined_sha256", None)
    body.setdefault("schema", V8_EXTERNAL_UNLOCK_SCHEMA)
    body.setdefault("protocol", V8_PROTOCOL_ID)
    return validate_v8_external_unlock_manifest(
        {**body, "combined_sha256": sha256_fingerprint(body)}
    )


def validate_v8_external_unlock_manifest(
    manifest: Mapping[str, Any] | str | Path,
    *,
    expected_source_tree_sha256: str | None = None,
    expected_parent_freeze_sha256: str | None = None,
) -> dict[str, Any]:
    value = read_json(manifest) if isinstance(manifest, (str, Path)) else dict(manifest)
    required = {
        "schema",
        "protocol",
        "created_at_utc",
        "source_tree_sha256",
        "parent_freeze_sha256",
        "architecture_adaptation",
        "training",
        "dataset",
        "analysis_plan",
        "evidence_hashes",
        "heldout_access",
        "combined_sha256",
    }
    if set(value) != required:
        raise V8FreezeManifestError(
            f"V8 external unlock fields must be exactly {sorted(required)}"
        )
    if (
        value["schema"] != V8_EXTERNAL_UNLOCK_SCHEMA
        or value["protocol"] != V8_PROTOCOL_ID
    ):
        raise V8FreezeManifestError("unsupported V8 external-unlock schema")
    if expected_source_tree_sha256 is not None and value[
        "source_tree_sha256"
    ] != str(expected_source_tree_sha256):
        raise V8FreezeManifestError("active source tree differs from external unlock")
    if expected_parent_freeze_sha256 is not None and value[
        "parent_freeze_sha256"
    ] != str(expected_parent_freeze_sha256):
        raise V8FreezeManifestError("external unlock references another architecture freeze")
    adaptation = value["architecture_adaptation"]
    if (
        not isinstance(adaptation, Mapping)
        or adaptation.get("permitted_change") != "four_class_head_to_binary_head_only"
        or int(adaptation.get("target_sfreq", 0)) != 250
        or len(adaptation.get("ordered_channels", ())) != 22
    ):
        raise V8FreezeManifestError("external architecture adaptation is not locked")
    dataset = value["dataset"]
    if (
        not isinstance(dataset, Mapping)
        or dataset.get("train_session") != "S1"
        or dataset.get("evaluation_session") != "S2"
        or not dataset.get("confirmatory_subjects")
    ):
        raise V8FreezeManifestError("external dataset contract is incomplete")
    heldout = value["heldout_access"]
    if (
        not isinstance(heldout, Mapping)
        or heldout.get("openbmi_s2_accessed_before_unlock") is not False
        or heldout.get("s2_checkpoint_selection_allowed") is not False
        or heldout.get("s2_gradient_updates_allowed") is not False
    ):
        raise V8FreezeManifestError("OpenBMI S2 was accessed or authorized for fitting")
    hashes = value["evidence_hashes"]
    if not isinstance(hashes, Mapping) or not hashes:
        raise V8FreezeManifestError("external unlock has no frozen evidence hashes")
    expected = sha256_fingerprint(
        {key: item for key, item in value.items() if key != "combined_sha256"}
    )
    if value["combined_sha256"] != expected:
        raise V8FreezeManifestError("external unlock digest does not match its contents")
    return value


def write_v8_external_unlock_manifest(
    path: str | Path, payload: Mapping[str, Any]
) -> Path:
    return write_json(path, build_v8_external_unlock_manifest(payload))


def _dataset_family(value: Any) -> str:
    normalized = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    if normalized in _BCI2A_ALIASES:
        return "bci2a"
    if normalized in _OPENBMI_ALIASES:
        return "openbmi"
    return normalized


def assert_v8_data_access(
    metadata: Any,
    *,
    stage: V8Stage,
    role: V8Role,
) -> bool:
    """Enforce the preregistered Session-T/E and OpenBMI-S1/S2 boundaries."""

    if stage not in {"development", "bci2a_evaluation", "openbmi_confirmation"}:
        raise V8DataAccessError(f"unknown V8 stage: {stage}")
    if role not in _FIT_ROLES | {"evaluation"}:
        raise V8DataAccessError(f"unknown V8 data role: {role}")
    rows = validate_trial_metadata(
        metadata,
        allowed_sessions=("T", "E", "S1", "S2"),
    )
    for row in rows:
        family = _dataset_family(row["dataset"])
        session = str(row["session"])
        if stage == "development":
            allowed = family == "bci2a" and session == "T" and role in _FIT_ROLES
        elif stage == "bci2a_evaluation":
            allowed = family == "bci2a" and (
                (role in _FIT_ROLES and session == "T")
                or (role == "evaluation" and session == "E")
            )
        else:
            allowed = family == "openbmi" and (
                (role in _FIT_ROLES and session == "S1")
                or (role == "evaluation" and session == "S2")
            )
        if not allowed:
            raise V8DataAccessError(
                "V8 data lock rejected "
                f"dataset={row['dataset']!r}, session={session!r}, stage={stage!r}, role={role!r}"
            )
    return True


def v8_heldout_lock_manifest() -> dict[str, Any]:
    """Return the machine-readable lock used before architecture freeze."""

    return {
        "protocol": V8_PROTOCOL_ID,
        "active_stage": "development",
        "allowed": [{"dataset": "bci2a", "session": "T", "roles": sorted(_FIT_ROLES)}],
        "locked": [
            {"dataset": "bci2a", "session": "E"},
            {"dataset": "openbmi", "session": "S2"},
        ],
        "unlock_requires": {
            "bci2a_E": "signed V8 architecture-freeze manifest",
            "openbmi_S2": "passed E6 audit, completed E7 utility evaluation, and signed external-confirmation manifest",
        },
    }


def assert_delay_control_contract(
    *,
    full_current: Any,
    zero_current: Any,
    full_delay_probability: Any,
    zero_delay_probability: Any,
    full_fractional_delay: Any,
    zero_fractional_delay: Any,
    full_routing_fingerprint: str,
    zero_routing_fingerprint: str,
    shared_state_before: Mapping[str, str],
    shared_state_after: Mapping[str, str],
    minimum_current_rms: float = 0.0,
) -> bool:
    """Check that full/zero differ only in their conditional lag transport.

    The function accepts NumPy arrays or tensors without importing either at
    module import time.  Formal runners must call it before interpreting a
    full-delay versus matched-zero comparison.
    """

    if dict(shared_state_before) != dict(shared_state_after):
        changed = sorted(
            key
            for key in set(shared_state_before) | set(shared_state_after)
            if shared_state_before.get(key) != shared_state_after.get(key)
        )
        raise V8ProtocolError(
            "shared model state changed during delay control: " + ", ".join(changed)
        )
    import numpy as np

    if str(full_routing_fingerprint) != str(zero_routing_fingerprint):
        raise V8ProtocolError("full and zero controls have different routing invariants")

    full = np.asarray(full_current)
    zero = np.asarray(zero_current)
    if full.shape != zero.shape or full.size == 0:
        raise V8ProtocolError("full and zero routed currents have incompatible shapes")
    if not np.isfinite(full).all() or not np.isfinite(zero).all():
        raise V8ProtocolError("full or zero routed current contains NaN or Inf")
    full_rms = float(np.sqrt(np.mean(np.square(np.abs(full)), dtype=np.float64)))
    zero_rms = float(np.sqrt(np.mean(np.square(np.abs(zero)), dtype=np.float64)))
    if full_rms <= float(minimum_current_rms) or zero_rms <= float(minimum_current_rms):
        raise V8ProtocolError(
            "matched full/zero controls must both carry routed current; "
            f"full_rms={full_rms:.3e}, zero_rms={zero_rms:.3e}"
        )

    full_probability = np.asarray(full_delay_probability, dtype=np.float64)
    zero_probability = np.asarray(zero_delay_probability, dtype=np.float64)
    full_fraction = np.asarray(full_fractional_delay, dtype=np.float64)
    zero_fraction = np.asarray(zero_fractional_delay, dtype=np.float64)
    if (
        full_probability.ndim != 2
        or full_probability.shape != zero_probability.shape
        or full_fraction.shape != zero_fraction.shape
        or full_fraction.shape != (full_probability.shape[0],)
    ):
        raise V8ProtocolError("matched delay operators have incompatible shapes")
    if not np.isfinite(full_probability).all() or not np.isfinite(zero_probability).all():
        raise V8ProtocolError("matched delay posterior contains NaN or Inf")
    if not np.allclose(full_probability.sum(axis=1), 1.0, rtol=0.0, atol=1e-6):
        raise V8ProtocolError("full delay posterior is not row-normalized")
    expected_zero = np.zeros_like(zero_probability)
    expected_zero[:, 0] = 1.0
    if not np.array_equal(zero_probability, expected_zero) or not np.array_equal(
        zero_fraction, np.zeros_like(zero_fraction)
    ):
        raise V8ProtocolError("matched-zero must be an exact point mass at integer lag zero")
    if np.array_equal(full_probability, zero_probability) and np.array_equal(
        full_fraction, zero_fraction
    ):
        raise V8ProtocolError("full and matched-zero delay operators are identical")
    return True


def mapping_sha256(state: Mapping[str, Any]) -> dict[str, str]:
    """Hash named tensor/array states without relying on object identity."""

    result: dict[str, str] = {}
    for name, value in sorted(state.items()):
        if hasattr(value, "detach"):
            value = value.detach().cpu().contiguous().numpy()
        if hasattr(value, "tobytes"):
            payload = value.tobytes()
        else:
            payload = canonical_json(value).encode("utf-8")
        result[str(name)] = hashlib.sha256(payload).hexdigest()
    return result


def validate_v8_hpo_budget(
    configurations: Sequence[Mapping[str, Any]],
    *,
    maximum: int = 24,
) -> bool:
    """Reject repeated or oversized architecture searches before execution."""

    if maximum < 1:
        raise V8ProtocolError("HPO maximum must be positive")
    if len(configurations) > int(maximum):
        raise V8ProtocolError(
            f"V8 HPO budget exceeded: {len(configurations)} configurations > {maximum}"
        )
    signatures = [sha256_fingerprint(dict(item)) for item in configurations]
    if len(signatures) != len(set(signatures)):
        raise V8ProtocolError("V8 HPO configurations contain duplicates")
    return True
