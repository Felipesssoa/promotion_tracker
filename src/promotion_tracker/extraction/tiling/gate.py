"""Filtros rápidos que decidem se uma imagem merece a leitura por recortes.

Três opções (escolhidas por --gate):
  flyer    (padrão) lê TODO encarte, mas só se for encarte. Regra central: sem
           preço na imagem, não é encarte. O OCR conta preços; só nos casos
           duvidosos pergunta ao modelo de visão (uma pergunta curta, com a
           imagem reduzida e o critério rigoroso do extrator original).
  products só deixa passar encarte com produto monitorado (o OCRAnalyzer do
           projeto, como o LLMExtractor antigo).
  none     sem filtro.
"""
from __future__ import annotations

import io
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class GateResult:
    passed: bool
    reason: str | None = None
    hits: tuple[str, ...] = ()  # produtos do segmento achados pelo OCR


class FlyerGate(Protocol):
    def check(self, image_path: Path) -> GateResult: ...


class AlwaysPassGate:
    """Sem filtro: toda imagem vai para a leitura por recortes."""

    def check(self, image_path: Path) -> GateResult:
        return GateResult(True)


# --------------------------------------------------------------------------- products
class OcrAnalyzerGate:
    """Só aprova imagem com produto monitorado (OCRAnalyzer do projeto)."""

    def __init__(self, analyzer: Any) -> None:
        self.analyzer = analyzer

    @classmethod
    def from_project(cls, debug: bool = False) -> "OcrAnalyzerGate":
        return cls(_build_analyzer(debug))

    def check(self, image_path: Path) -> GateResult:
        analysis = self.analyzer.analyze(image_path)
        if analysis is None:
            return GateResult(False, self.analyzer.ignored_reason or "sem_produto_do_segmento")
        return GateResult(True, None, tuple(analysis.found_products))


def _build_analyzer(debug: bool = False) -> Any:
    from promotion_tracker.common.products import ProductCatalog
    from promotion_tracker.extraction.ocr.analyzer import OCRAnalyzer

    catalog = ProductCatalog()
    return OCRAnalyzer(
        catalog.products,
        debug=debug,
        product_aliases=catalog.search_terms_by_product,
    )


# --------------------------------------------------------------------------- flyer
_RE_FULL_PRICE = re.compile(r"(?<!\d)\d{1,4}\s*[.,]\s*\d{2}(?!\d)")
_RE_REAIS = re.compile(r"r\$\s*\d")
_RE_BIG_DIGITS = re.compile(r"\d{1,4}")
_RE_CENTS = re.compile(r"[.,]?\s*\d{2}")

# Palavras que reforçam um encarte (já sem acento e em minúsculas). Só reforçam:
# a decisão central é a presença de preço.
_KEYWORDS = (
    "oferta", "ofertas", "promocao", "promocoes", "validade", "valido", "leve",
    "pague", "economize", "estoque", "clube", "atacado", "tabloide", "imperdivel",
    "aproveite",
)
_KEYWORD_PATTERNS = [re.compile(rf"\b{w}\b") for w in _KEYWORDS]

CLASSIFY_PROMPT = (
    "Você analisa imagens de supermercados brasileiros. Responda encarte=true "
    "SOMENTE se a imagem mostrar, em formato de material promocional de "
    "supermercado, produtos com PREÇO em reais (R$ ou números grandes de preço) "
    "ao lado de cada produto. Um recorte de encarte também vale. Responda "
    "encarte=false para: foto de produto, embalagem ou prateleira sem etiqueta "
    "de preço; foto pessoal, paisagem, comida, receita; logo, banner "
    "institucional, anúncio de marca, cupom, print de rede social, meme, texto "
    "solto; imagem sem nenhum preço legível. Na dúvida, encarte=false. "
    "Responda somente com JSON."
)
CLASSIFY_SCHEMA = {
    "type": "object",
    "properties": {"encarte": {"type": "boolean"}},
    "required": ["encarte"],
}


def _count_split_prices(lines: list[Any]) -> int:
    """Preços escritos com número grande e centavos pequenos ("5" + "99").

    Só conta com as posições do OCR: um número de 1-4 dígitos com um bloco de
    2 dígitos MENOR (no máximo 80% da altura) logo à direita, alinhado ao topo.
    Números soltos do mesmo tamanho ("12 34 56") não formam preço.
    """
    big = [l for l in lines if _RE_BIG_DIGITS.fullmatch(l.text.strip())]
    cents = [l for l in lines if _RE_CENTS.fullmatch(l.text.strip())]
    used: set[int] = set()
    count = 0
    for number in big:
        height = max(1.0, number.y_end - number.y0)
        for index, small in enumerate(cents):
            if index in used or small is number:
                continue
            small_height = small.y_end - small.y0
            gap = small.x0 - number.x1
            if (
                small_height <= 0.8 * height
                and -0.2 * height <= gap <= 1.5 * height
                and abs(small.y0 - number.y0) <= 0.8 * height
            ):
                used.add(index)
                count += 1
                break
    return count


class FlyerDetector:
    """Decide "é encarte?" pela presença de preços, da camada mais barata à mais cara.

    Regra central: sem preço, não é encarte.
      0 preços                         -> não é encarte (sem chamar o modelo)
      muitos preços, ou preços + palavras de oferta -> encarte (só OCR)
      poucos preços (1, ou 2+ sem palavra de oferta) -> pergunta curta ao modelo,
                                          com o critério rigoroso do extrator original
    No fim, a leitura por recortes ainda confirma (sem oferta com preço = não encarte).
    """

    def __init__(
        self,
        analyzer: Any,
        vlm: Any | None = None,
        sure_prices: int = 4,
        min_prices_with_keywords: int = 2,
        fallback_min_prices: int = 2,
        vlm_long_side: int = 768,
    ) -> None:
        self.analyzer = analyzer
        self.vlm = vlm
        self.sure_prices = sure_prices
        self.min_prices_with_keywords = min_prices_with_keywords
        self.fallback_min_prices = fallback_min_prices
        self.vlm_long_side = vlm_long_side

    @classmethod
    def from_project(cls, vlm: Any | None = None, **kwargs: Any) -> "FlyerDetector":
        return cls(_build_analyzer(), vlm=vlm, **kwargs)

    # -- camada 1: OCR
    def _read(self, image_path: Path) -> tuple[list[str], list[Any]]:
        import numpy as np
        from PIL import Image

        self.analyzer._load_ocr()  # carrega o motor uma vez só
        with Image.open(image_path) as image:
            rgb = np.array(image.convert("RGB"))
        texts, lines, _positions = self.analyzer._read_ocr(rgb)
        return texts, lines

    @staticmethod
    def _signals(texts: list[str], lines: list[Any]) -> tuple[int, int]:
        from promotion_tracker.extraction.ocr.matching import normalize_search_text

        complete = sum(
            1 for t in texts
            if _RE_FULL_PRICE.search(t) or _RE_REAIS.search(t.lower().replace(" ", ""))
        )
        prices = complete + _count_split_prices(lines)
        haystack = normalize_search_text(" ".join(texts))
        keywords = sum(1 for pattern in _KEYWORD_PATTERNS if pattern.search(haystack))
        return prices, keywords

    # -- camada 2: pergunta curta ao modelo
    def _ask_model(self, image_path: Path) -> bool | None:
        from PIL import Image

        try:
            with Image.open(image_path) as image:
                image = image.convert("RGB")
                image.thumbnail((self.vlm_long_side, self.vlm_long_side))
                buffer = io.BytesIO()
                image.save(buffer, format="JPEG", quality=85)
            raw = self.vlm.extract(
                buffer.getvalue(), CLASSIFY_PROMPT, schema=CLASSIFY_SCHEMA, max_tokens=24
            )
            match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
            return bool(json.loads(match.group(0) if match else raw).get("encarte"))
        except Exception as exc:
            log.warning("Pergunta ao modelo falhou (%s); usando a regra simples.", exc)
            return None

    def check(self, image_path: Path) -> GateResult:
        texts, lines = self._read(image_path)
        prices, keywords = self._signals(texts, lines)
        hits = tuple(self.analyzer._find_products(texts))
        summary = f"{prices} preços, {keywords} palavras de oferta"

        if prices == 0:
            log.info("Detector: %s -> não é encarte (sem preço)", summary)
            return GateResult(False, "sem_preco_na_imagem")

        if prices >= self.sure_prices or (
            prices >= self.min_prices_with_keywords and keywords >= 1
        ):
            log.info("Detector: %s -> encarte (OCR)", summary)
            return GateResult(True, None, hits)

        # Poucos preços e nenhum reforço: um card de oferta única ou um falso positivo.
        if self.vlm is not None:
            answer = self._ask_model(image_path)
            if answer is not None:
                log.info("Detector: %s -> %s (modelo)", summary,
                         "encarte" if answer else "não é encarte")
                return GateResult(answer, None if answer else "nao_e_encarte_modelo", hits)

        passed = prices >= self.fallback_min_prices
        log.info("Detector: %s -> %s (regra simples)", summary,
                 "encarte" if passed else "não é encarte")
        return GateResult(passed, None if passed else "poucos_precos_na_imagem", hits)