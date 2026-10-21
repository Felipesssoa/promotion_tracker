"""Normaliza marcas e produtos da Silver para a camada Silver reviewed."""

from __future__ import annotations

import difflib
import json
import logging
from pathlib import Path
from typing import Any

import yaml

from ..common.json_io import write_json_atomic
from ..common.paths import PROJECT_PATHS
from ..common.products import ProductCatalog, normalize_product_text

logger = logging.getLogger(__name__)


class Normalizer:
    """Gera JSONs revisados sem modificar os arquivos originais da Silver."""

    def __init__(
        self,
        silver_dir: Path | None = None,
        reviewed_dir: Path | None = None,
        brands_path: Path | None = None,
        product_catalog: ProductCatalog | None = None,
    ) -> None:
        self.silver_dir = silver_dir or PROJECT_PATHS.silver_dir
        self.reviewed_dir = reviewed_dir or PROJECT_PATHS.silver_reviewed_dir
        self.brands_path = brands_path or PROJECT_PATHS.brands_config
        self.product_catalog = product_catalog or ProductCatalog()
        self.brand_aliases = self._load_brands()

    def _load_brands(self) -> dict[str, str]:
        if not self.brands_path.is_file():
            raise FileNotFoundError(
                f"Arquivo de marcas não encontrado: {self.brands_path}"
            )
        try:
            configuration = yaml.safe_load(
                self.brands_path.read_text(encoding="utf-8")
            )
        except yaml.YAMLError as exc:
            raise ValueError(
                f"YAML inválido em {self.brands_path}: {exc}"
            ) from exc
        entries = (
            configuration.get("marcas")
            if isinstance(configuration, dict) else None
        )
        if not isinstance(entries, list):
            raise ValueError(
                f"A chave 'marcas' deve ser uma lista em {self.brands_path}"
            )

        aliases: dict[str, str] = {}
        for index, entry in enumerate(entries):
            if not isinstance(entry, dict) or not isinstance(
                entry.get("nome_oficial"), str
            ):
                raise ValueError(
                    f"Item {index} inválido em {self.brands_path}: "
                    "informe 'nome_oficial' e 'aliases'."
                )
            official_name = entry["nome_oficial"].strip()
            brand_aliases = entry.get("aliases", [])
            if not isinstance(brand_aliases, list) or not all(
                isinstance(alias, str) for alias in brand_aliases
            ):
                raise ValueError(
                    f"'aliases' deve ser uma lista de textos no item {index} "
                    f"de {self.brands_path}."
                )
            for alias in [official_name, *brand_aliases]:
                normalized_alias = normalize_product_text(alias)
                if normalized_alias:
                    aliases.setdefault(normalized_alias, official_name)
        if not aliases:
            raise ValueError(f"Nenhuma marca válida em {self.brands_path}")
        return aliases

    def _review_brand(self, brand: str | None) -> str | None:
        if brand is None:
            return None
        normalized_brand = normalize_product_text(brand)
        if not normalized_brand:
            return brand
        exact_match = self.brand_aliases.get(normalized_brand)
        if exact_match:
            return exact_match

        candidates = difflib.get_close_matches(
            normalized_brand,
            self.brand_aliases.keys(),
            n=2,
            cutoff=0.84,
        )
        if not candidates:
            return brand
        best_match = candidates[0]
        best_score = difflib.SequenceMatcher(
            None, normalized_brand, best_match
        ).ratio()
        if len(candidates) > 1:
            second_match = candidates[1]
            second_score = difflib.SequenceMatcher(
                None, normalized_brand, second_match
            ).ratio()
            if (
                self.brand_aliases[best_match] != self.brand_aliases[second_match]
                and best_score - second_score < 0.04
            ):
                logger.info("Marca ambígua mantida sem revisão: %s", brand)
                return brand
        return self.brand_aliases[best_match]

    def normalize(self) -> int:
        """Processa todos os JSONs da Silver e retorna a quantidade gerada."""
        if not self.silver_dir.is_dir():
            raise FileNotFoundError(f"Silver inexistente: {self.silver_dir}")

        generated = 0
        source_paths = sorted(self.silver_dir.rglob("*.json"))
        source_files = {
            source_path.relative_to(self.silver_dir)
            for source_path in source_paths
        }
        for source_path in source_paths:
            relative_path = source_path.relative_to(self.silver_dir)
            output_path = self.reviewed_dir / relative_path
            try:
                record = json.loads(source_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                logger.error("Falha ao ler JSON da Silver %s: %s", source_path, exc)
                raise ValueError(
                    f"Não foi possível normalizar {source_path}"
                ) from exc
            if not isinstance(record, dict):
                raise ValueError(
                    f"A raiz do JSON deve ser um objeto: {source_path}"
                )

            self._normalize_record(record)
            write_json_atomic(output_path, record)
            generated += 1

        if self.reviewed_dir.is_dir():
            for reviewed_path in self.reviewed_dir.rglob("*.json"):
                if reviewed_path.relative_to(self.reviewed_dir) not in source_files:
                    reviewed_path.unlink()
                    logger.info("JSON revisado obsoleto removido: %s", reviewed_path)

        logger.info(
            "Silver reviewed: %d arquivo(s) gerado(s) em %s",
            generated,
            self.reviewed_dir,
        )
        return generated

    def _normalize_record(self, record: dict[str, Any]) -> None:
        products = record.get("produtos", [])
        if not isinstance(products, list):
            raise ValueError("O campo 'produtos' deve ser uma lista.")

        for item in products:
            if not isinstance(item, dict):
                logger.warning("Item de produto inválido mantido sem alteração.")
                continue

            original_product = item.get("produto_original", item.get("produto"))
            reviewed_product = self.product_catalog.normalize_name(
                original_product if isinstance(original_product, str) else None
            )
            item["produto_original"] = original_product
            item["produto_revisado"] = reviewed_product
            item["produto"] = reviewed_product

            original_brand = item.get("marca_original", item.get("marca"))
            reviewed_brand = self._review_brand(
                original_brand if isinstance(original_brand, str) else None
            )
            item["marca_original"] = original_brand
            item["marca_revisada"] = reviewed_brand
            item["marca"] = reviewed_brand

        found_products = record.get("produtos_encontrados")
        if isinstance(found_products, list):
            record["produtos_encontrados"] = [
                self.product_catalog.normalize_name(product)
                if isinstance(product, str) else product
                for product in found_products
            ]
