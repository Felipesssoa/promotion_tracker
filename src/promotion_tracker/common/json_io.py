"""Rotinas compartilhadas de persistência JSON."""

import json
from pathlib import Path
from typing import Any


def write_json_atomic(output_path: Path, data: dict[str, Any]) -> None:
    """Grava JSON UTF-8 em um temporário e substitui o arquivo de destino."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    try:
        temp_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=4),
            encoding="utf-8",
        )
        temp_path.replace(output_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()
