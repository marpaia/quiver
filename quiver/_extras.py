"""Friendly loaders for the optional-dependency extras (``sim``, ``data``).

Importing an optional backend through :func:`require` turns a bare ``ModuleNotFoundError`` deep in a call
into a message that names the extra to install.
"""

from __future__ import annotations

import importlib
from types import ModuleType


def require(module: str, extra: str) -> ModuleType:
    """Import ``module``, or raise a ``ModuleNotFoundError`` naming the extra that provides it."""
    try:
        return importlib.import_module(module)
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            f"{module} is required here; install the '{extra}' extra (uv sync --extra {extra})."
        ) from exc
