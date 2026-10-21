"""Normalização e correspondência de nomes de produtos lidos pelo OCR.

O pipeline normaliza tanto a lista de referência quanto o texto reconhecido
antes de comparar. A correspondência tenta primeiro termos completos e,
depois, texto sem espaços e similaridade aproximada, quando habilitados.
"""

import re
import unicodedata

from ..settings import (
    MATCH_FUZZY,
    MATCH_JOINED_TEXT,
    FUZZY_THRESHOLD,
    MIN_TOLERANT_CHARS,
)

try:
    from rapidfuzz import fuzz  # type: ignore
except Exception:  # rapidfuzz é opcional
    fuzz = None

FUZZY_AVAILABLE = fuzz is not None


def normalize_search_text(text: str) -> str:
    """Padroniza texto para comparações sem distinção de caixa ou acento.

    Mantém letras e números, substitui pontuação por espaços e comprime
    espaços repetidos. O resultado é adequado às funções de casamento deste
    módulo, mas não deve ser usado como texto de apresentação.

    Args:
        text: Texto original reconhecido ou nome de produto.

    Returns:
        Texto em minúsculas, sem acentos e com palavras separadas por espaço.
    """
    text = unicodedata.normalize("NFKD", text.casefold())
    text = "".join(
        character for character in text
        if not unicodedata.combining(character)
    )
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text).split())


def product_matches(
    product: str, normalized_text: str, allow_unordered_words: bool = False
) -> bool:
    """Verifica se um nome de produto aparece no texto normalizado do OCR.

    A comparação segue esta ordem: frase completa, palavras independentes
    (opcional), texto colado e similaridade aproximada. As comparações
    tolerantes são limitadas por tamanho mínimo e configuração para reduzir
    falsos positivos.

    Args:
        product: Nome do produto já normalizado.
        normalized_text: Texto do OCR já normalizado.
        allow_unordered_words: Aceita todas as palavras do produto em qualquer
            ordem quando ``True``; útil para linhas/colunas reorganizadas.

    Returns:
        ``True`` se o produto corresponder ao texto; caso contrário, ``False``.
    """
    if not product or not normalized_text:
        return False

    if f" {product} " in f" {normalized_text} ":
        return True

    if allow_unordered_words:
        terms = product.split()
        if len(terms) > 1:
            words = set(normalized_text.split())
            if all(term in words for term in terms):
                return True

    compact_text = product.replace(" ", "")
    if len(compact_text) < MIN_TOLERANT_CHARS:
        return False

    if MATCH_JOINED_TEXT and compact_text in normalized_text.replace(" ", ""):
        return True

    if MATCH_FUZZY and fuzz is not None:
        return fuzz.partial_ratio(product, normalized_text) >= FUZZY_THRESHOLD

    return False