"""Catálogo e normalização dos produtos monitorados."""

from __future__ import annotations

import difflib
import re
import unicodedata
from pathlib import Path
from typing import Any

import yaml

from .paths import PROJECT_PATHS


def normalize_product_text(text: str) -> str:
    """Normaliza texto para comparação de produtos sem acentos ou pontuação."""
    normalized = unicodedata.normalize("NFKD", text.casefold())
    without_accents = "".join(
        character for character in normalized
        if not unicodedata.combining(character)
    )
    return " ".join(re.sub(r"[^a-z0-9]+", " ", without_accents).split())


class ProductCatalog:
    """Carrega os produtos válidos e seus termos equivalentes."""

    def __init__(
        self,
        products_path: Path | None = None,
        aliases_path: Path | None = None,
    ) -> None:
        self.products_path = products_path or PROJECT_PATHS.product_config
        self.aliases_path = aliases_path or PROJECT_PATHS.product_aliases_config
        self.products = self._load_products()
        self.search_terms_by_product = self._load_aliases()
        self._term_to_product = {
            normalize_product_text(term): product
            for product, terms in self.search_terms_by_product.items()
            for term in terms
        }
        for product in self.products:
            self._term_to_product[normalize_product_text(product)] = product

    def _load_yaml(self, path: Path) -> dict[str, Any]:
        if not path.is_file():
            raise FileNotFoundError(f"Arquivo de configuração não encontrado: {path}")
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise ValueError(f"YAML inválido em {path}: {exc}") from exc
        if not isinstance(data, dict):
            raise ValueError(f"A raiz de {path} deve ser um objeto YAML.")
        return data

    def _load_products(self) -> list[str]:
        entries = self._load_yaml(self.products_path).get("produtos")
        if not isinstance(entries, list):
            raise ValueError(
                f"A chave 'produtos' deve ser uma lista em {self.products_path}"
            )
        products = list(dict.fromkeys(
            item["produto"].strip()
            for item in entries
            if isinstance(item, dict)
            and isinstance(item.get("produto"), str)
            and item["produto"].strip()
        ))
        if not products:
            raise ValueError(f"Nenhum produto válido em {self.products_path}")
        return products

    def _load_aliases(self) -> dict[str, list[str]]:
        entries = self._load_yaml(self.aliases_path).get("abreviacoes")
        if not isinstance(entries, list):
            raise ValueError(
                f"A chave 'abreviacoes' deve ser uma lista em {self.aliases_path}"
            )

        products_by_key = {
            normalize_product_text(product): product for product in self.products
        }
        aliases = {
            product: [product]
            for product in self.products
        }
        for index, entry in enumerate(entries):
            if not isinstance(entry, dict) or not isinstance(
                entry.get("produto"), str
            ):
                raise ValueError(
                    f"Item {index} inválido em {self.aliases_path}: "
                    "informe 'produto' e 'termos'."
                )
            product_key = normalize_product_text(entry["produto"])
            canonical_product = products_by_key.get(product_key)
            if canonical_product is None:
                raise ValueError(
                    f"O produto {entry['produto']!r} em {self.aliases_path} "
                    f"não existe em {self.products_path}."
                )
            terms = entry.get("termos")
            if not isinstance(terms, list) or not all(
                isinstance(term, str) and term.strip() for term in terms
            ):
                raise ValueError(
                    f"'termos' deve ser uma lista de textos no item "
                    f"{index} de {self.aliases_path}."
                )
            aliases[canonical_product].extend(terms)

        return {
            product: list(dict.fromkeys(terms))
            for product, terms in aliases.items()
        }

    def normalize_name(self, name: str | None) -> str | None:
        """Converte alias ou abreviação reconhecida para o produto canônico."""
        if name is None:
            return None
        normalized_name = normalize_product_text(name)
        if not normalized_name:
            return name

        exact_product = self._term_to_product.get(normalized_name)
        if exact_product:
            return exact_product

        padded_name = f" {normalized_name} "
        matching_terms = [
            (normalize_product_text(term), product)
            for product, terms in self.search_terms_by_product.items()
            for term in terms
            if f" {normalize_product_text(term)} " in padded_name
        ]
        if matching_terms:
            matching_terms.sort(key=lambda match: len(match[0]), reverse=True)
            return matching_terms[0][1]

        candidates = difflib.get_close_matches(
            normalized_name,
            self._term_to_product.keys(),
            n=2,
            cutoff=0.86,
        )
        if not candidates:
            return name
        if (
            len(candidates) > 1
            and self._term_to_product[candidates[0]]
            != self._term_to_product[candidates[1]]
            and difflib.SequenceMatcher(
                None, normalized_name, candidates[0]
            ).ratio()
            - difflib.SequenceMatcher(
                None, normalized_name, candidates[1]
            ).ratio()
            < 0.04
        ):
            return name
        return self._term_to_product[candidates[0]]
