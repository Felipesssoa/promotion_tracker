"""Junta ofertas repetidas na zona de sobreposição entre recortes vizinhos.

Regra de ouro: ofertas do MESMO recorte nunca são fundidas entre si. Assim,
dois produtos legítimos iguais no mesmo encarte (ex.: dois sabores com o mesmo
preço) não se perdem; só se funde o que veio de recortes diferentes.
"""
from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher
from typing import Any

from promotion_tracker.extraction.llm.schemas import Oferta

try:  # opcional, mais rápido e melhor
    from rapidfuzz import fuzz as _fuzz
except ImportError:  # pragma: no cover
    _fuzz = None

_UNIT_MAP = {"gr": "g", "grs": "g", "lt": "l", "lts": "l", "litro": "l", "litros": "l",
             "und": "un", "unid": "un", "unidade": "un"}


def normalize(text: str | None) -> str:
    text = unicodedata.normalize("NFKD", text or "")
    text = "".join(c for c in text if not unicodedata.combining(c)).lower()
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def similarity(a: str | None, b: str | None) -> float:
    na, nb = normalize(a), normalize(b)
    if not na or not nb:
        return 0.0
    if _fuzz is not None:
        return _fuzz.token_sort_ratio(na, nb) / 100
    ta, tb = " ".join(sorted(na.split())), " ".join(sorted(nb.split()))
    return SequenceMatcher(None, ta, tb).ratio()


def _unit(value: str | None) -> str:
    cleaned = normalize(value)
    return _UNIT_MAP.get(cleaned, cleaned)


def _prices(offer: Oferta) -> set[float]:
    return {p for p in (offer.preco_atual, offer.preco_anterior, offer.preco_condicao)
            if p is not None}


def _names(offer: Oferta) -> str:
    return " | ".join(sorted(normalize(v.produto) for v in offer.variantes if v.produto))


def _match_score(a: Oferta, b: Oferta, threshold: float) -> float | None:
    """Similaridade dos nomes se as ofertas parecem a mesma; senão None."""
    pa, pb = _prices(a), _prices(b)
    # Compatíveis se algum preço coincide: tolera campos trocados entre recortes.
    if pa and pb and not any(abs(x - y) <= 0.005 for x in pa for y in pb):
        return None

    if len(a.variantes) == 1 and len(b.variantes) == 1:
        va, vb = a.variantes[0], b.variantes[0]
        if va.quantidade is not None and vb.quantidade is not None \
                and abs(va.quantidade - vb.quantidade) > 1e-6:
            return None
        ua, ub = _unit(va.unidade_medida), _unit(vb.unidade_medida)
        if ua and ub and ua != ub and "outro" not in (ua, ub):
            return None
        if va.marca and vb.marca and similarity(va.marca, vb.marca) < 0.8:
            return None

    na, nb = _names(a), _names(b)
    if not na or not nb:
        return None
    score = similarity(na, nb)
    if na in nb or nb in na:  # leitura parcial de um nome cortado
        score = max(score, 0.9)
    return score if score >= threshold else None


def _quality(offer: Oferta) -> int:
    filled = sum(x is not None for x in (
        offer.preco_atual, offer.preco_anterior, offer.preco_condicao, offer.condicao))
    for v in offer.variantes:
        filled += sum(x not in (None, "") for x in (v.produto, v.marca, v.quantidade, v.unidade_medida))
    return filled + len(offer.variantes)


def merge_offers(items: list[dict[str, Any]], threshold: float = 0.85) -> list[Oferta]:
    """`items`: [{"oferta": Oferta, "tile": int}, ...] na ordem de leitura."""
    clusters: list[dict[str, Any]] = []

    for item in items:
        best, best_score = None, 0.0
        for cluster in clusters:
            if item["tile"] in cluster["tiles"]:
                continue
            score = _match_score(cluster["oferta"], item["oferta"], threshold)
            if score is not None and score > best_score:
                best, best_score = cluster, score

        if best is None:
            clusters.append({"oferta": item["oferta"], "tiles": {item["tile"]}})
            continue
        best["tiles"].add(item["tile"])
        if _quality(item["oferta"]) > _quality(best["oferta"]):
            best["oferta"] = item["oferta"]

    return [cluster["oferta"] for cluster in clusters]