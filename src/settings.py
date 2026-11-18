"""Rutas del proyecto y lectura de config/config.yaml (compartido por main y scripts)."""
from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "config.yaml"


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def project_path(p: str | Path) -> Path:
    """Rutas relativas de la config se interpretan desde la raíz del repo."""
    p = Path(p)
    return p if p.is_absolute() else ROOT / p
