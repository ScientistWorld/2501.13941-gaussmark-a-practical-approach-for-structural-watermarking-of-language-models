"""Shared path defaults for the GaussMark reproduction."""

from __future__ import annotations

import os
from pathlib import Path


MISTRAL_RELATIVE_PATH = "mistralai/Mistral-7B-Instruct-v0.3"
C4_RELATIVE_PATH = "c4_realnewslike"
REPO_ROOT = Path(__file__).resolve().parents[1]
DOWNLOAD_ROOT = REPO_ROOT / "data" / "downloads"


def _first_existing(candidates: list[Path], sentinel: str) -> Path:
    for path in candidates:
        if (path / sentinel).exists():
            return path
    return candidates[0]


def default_model_path() -> str:
    """Return the preferred local Mistral model path."""
    env_path = os.environ.get("GAUSSMARK_MODEL_PATH")
    if env_path:
        return env_path
    return str(
        _first_existing(
            [
                DOWNLOAD_ROOT / "models" / MISTRAL_RELATIVE_PATH,
                Path("/home/user/models") / MISTRAL_RELATIVE_PATH,
            ],
            "config.json",
        )
    )


def default_data_path() -> str:
    """Return the preferred local C4 realnewslike prompt path."""
    env_path = os.environ.get("GAUSSMARK_DATA_PATH")
    if env_path:
        return env_path
    return str(
        _first_existing(
            [
                DOWNLOAD_ROOT / C4_RELATIVE_PATH,
                Path("/home/user/data") / C4_RELATIVE_PATH,
            ],
            "examples.json",
        )
    )
