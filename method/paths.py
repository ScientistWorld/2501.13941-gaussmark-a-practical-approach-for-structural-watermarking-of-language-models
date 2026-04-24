"""Shared path defaults for the GaussMark reproduction."""

from __future__ import annotations

import os
from pathlib import Path


MISTRAL_RELATIVE_PATH = "mistralai/Mistral-7B-Instruct-v0.3"
C4_RELATIVE_PATH = "c4_realnewslike"
REPO_ROOT = Path(__file__).resolve().parents[1]
DOWNLOAD_ROOT = REPO_ROOT / "data" / "downloads"
LEGACY_MODEL_ROOT = REPO_ROOT / "models"
LEGACY_DATA_ROOT = REPO_ROOT / "data"


def _resolve_existing(preferred: Path, fallback: Path) -> str:
    """Prefer the portable download path, but tolerate older workspace layouts."""
    if preferred.exists():
        return str(preferred)
    if fallback.exists():
        return str(fallback)
    return str(preferred)

def default_model_path() -> str:
    """Return the preferred local Mistral model path."""
    env_path = os.environ.get("GAUSSMARK_MODEL_PATH")
    if env_path:
        return env_path
    return _resolve_existing(
        DOWNLOAD_ROOT / "models" / MISTRAL_RELATIVE_PATH,
        LEGACY_MODEL_ROOT / MISTRAL_RELATIVE_PATH,
    )


def default_data_path() -> str:
    """Return the preferred local C4 realnewslike prompt path."""
    env_path = os.environ.get("GAUSSMARK_DATA_PATH")
    if env_path:
        return env_path
    return _resolve_existing(
        DOWNLOAD_ROOT / C4_RELATIVE_PATH,
        LEGACY_DATA_ROOT / C4_RELATIVE_PATH,
    )
