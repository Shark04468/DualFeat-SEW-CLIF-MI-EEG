"""Optional dependency helpers."""

from __future__ import annotations

import importlib
import contextlib
import io
from types import ModuleType


def optional_import(name: str, purpose: str | None = None) -> ModuleType:
    try:
        return importlib.import_module(name)
    except ImportError as exc:
        msg = f"Optional dependency {name!r} is required"
        if purpose:
            msg += f" for {purpose}"
        msg += ". Install with: pip install -e \".[all]\""
        raise ImportError(msg) from exc


def has_module(name: str) -> bool:
    try:
        with contextlib.redirect_stderr(io.StringIO()):
            importlib.import_module(name)
        return True
    except Exception:
        return False
