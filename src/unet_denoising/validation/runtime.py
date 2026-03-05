from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

from unet_denoising.exceptions import ConfigValidationError



def ensure_dir_exists(path: str | Path, label: str) -> Path:
    p = Path(path)
    if not p.exists() or not p.is_dir():
        raise ConfigValidationError(f"{label} does not exist or is not a directory: {p}")
    return p



def ensure_file_exists(path: str | Path, label: str) -> Path:
    p = Path(path)
    if not p.exists() or not p.is_file():
        raise ConfigValidationError(f"{label} does not exist or is not a file: {p}")
    return p



def ensure_writable_dir(path: str | Path, label: str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    if not os.access(p, os.W_OK):
        raise ConfigValidationError(f"{label} is not writable: {p}")
    return p


def ensure_env_vars(names: Iterable[str], context: str) -> None:
    missing = [n for n in names if not os.getenv(n)]
    if missing:
        joined = ", ".join(missing)
        raise ConfigValidationError(f"Missing required environment variables for {context}: {joined}")
