from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def initialize_metrics_file(path: str | Path) -> Path:
    """Create or truncate a JSONL metrics file and return its path."""
    metrics_path = Path(path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text("", encoding="utf-8")
    return metrics_path


def append_metrics_record(path: str | Path, record: Mapping[str, Any]) -> None:
    """Append one JSON-serializable metrics record to a JSONL file."""
    metrics_path = Path(path)
    with metrics_path.open("a", encoding="utf-8") as metrics_file:
        metrics_file.write(json.dumps(dict(record), sort_keys=True) + "\n")
