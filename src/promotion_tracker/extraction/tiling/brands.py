"""Marcas conhecidas (config/brands.yaml): lista para o prompt e preenchimento de marca ausente."""
from __future__ import annotations

import logging
from pathlib import Path

import yaml

from promotion_tracker.common.paths import PROJECT_PATHS
from promotion_tracker.common.products import normalize_product_text

log = logging.getLogger(__name__)

_MIN_ALIAS_CHARS = 3


class BrandCatalog:
    def __init__(self, brands_path: Path | None = None) -> None:
        path = Path(brands_path) if brands_path else PROJECT_PATHS.brands_config
        self.official: list[str] = []
        self._aliases: list[tuple[str, str]] = []  # (alias normalizado, nome oficial)

        if not path.is_file():
            log.warning("brands.yaml não encontrado: %s", path)
            return
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        for entry in data.get("marcas", []) if isinstance(data, dict) else []:
            if not isinstance(entry, dict) or not isinstance(entry.get("nome_oficial"), str):
                continue
            official = entry["nome_oficial"].strip()
            self.official.append(official)
            for alias in [official, *(entry.get("aliases") or [])]:
                normalized = normalize_product_text(str(alias))
                if len(normalized) >= _MIN_ALIAS_CHARS:
                    self._aliases.append((normalized, official))
        # alias mais longo primeiro: "vale do sertao" antes de "vale"
        self._aliases.sort(key=lambda pair: len(pair[0]), reverse=True)
        log.info("Marcas conhecidas carregadas: %d", len(self.official))

    def prompt_list(self) -> str:
        return ", ".join(self.official)

    def find_in_text(self, text: str | None) -> str | None:
        """Marca oficial citada no texto (palavra inteira), ou None."""
        padded = f" {normalize_product_text(text or '')} "
        for alias, official in self._aliases:
            if f" {alias} " in padded:
                return official
        return None