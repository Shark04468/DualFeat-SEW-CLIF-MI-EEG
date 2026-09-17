"""Runtime cache placement helpers.

The cloud instances used for the full experiments often have a small root
filesystem and a larger data disk mounted at ``/root/autodl-tmp``. This helper
keeps heavyweight dataset and framework caches away from the root disk when
that data disk is available, while remaining a no-op on ordinary local setups.
"""

from __future__ import annotations

import os
from pathlib import Path


def default_storage_root() -> Path | None:
    explicit = os.environ.get("DPC_SNN_STORAGE_ROOT")
    if explicit:
        return Path(explicit).expanduser()
    autodl_tmp = Path("/root/autodl-tmp")
    if autodl_tmp.exists() and os.access(autodl_tmp, os.W_OK):
        return autodl_tmp / "DPC-SNN_storage"
    return None


def configure_cache_env(storage_root: str | Path | None = None) -> Path | None:
    root = Path(storage_root).expanduser() if storage_root is not None else default_storage_root()
    if root is None:
        return None
    cache = root / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    defaults = {
        "DPC_SNN_STORAGE_ROOT": str(root),
        "MNE_DATA": str(cache / "mne_data"),
        "MNE_DATASETS_EEGBCI_PATH": str(cache / "mne_data" / "MNE-eegbci-data"),
        "MOABB_DATA": str(cache / "moabb"),
        "XDG_CACHE_HOME": str(cache / "xdg"),
        "HF_HOME": str(cache / "hf"),
        "HF_DATASETS_CACHE": str(cache / "huggingface" / "datasets"),
        "TORCH_HOME": str(cache / "torch"),
        "MPLCONFIGDIR": str(cache / "matplotlib"),
        "PIP_CACHE_DIR": str(cache / "pip"),
        "TMPDIR": str(cache / "tmp"),
        "PYTHONPYCACHEPREFIX": str(root / "pycache"),
        "WANDB_DIR": str(root / "logs" / "wandb"),
    }
    for key, value in defaults.items():
        os.environ.setdefault(key, value)
        Path(os.environ[key]).mkdir(parents=True, exist_ok=True)
    return root
