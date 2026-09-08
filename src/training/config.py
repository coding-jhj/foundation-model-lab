from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


REQUIRED_SECTIONS = ("project", "data", "model", "training", "evaluation")


def load_config(path: str | Path) -> dict[str, Any]:
    """Load and minimally validate an experiment YAML configuration."""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    if not isinstance(config, dict):
        raise ValueError("configuration must contain a top-level mapping")

    missing_sections = [section for section in REQUIRED_SECTIONS if section not in config]
    if missing_sections:
        missing = ", ".join(missing_sections)
        raise ValueError(f"configuration is missing required sections: {missing}")

    return config
