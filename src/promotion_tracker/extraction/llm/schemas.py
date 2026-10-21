"""Contratos e normalização das respostas estruturadas do modelo.

Fluxo: o Ollama devolve JSON; os modelos Pydantic convertem tipos e limpam
campos; a validade é interpretada como data ISO antes de chegar ao resultado
persistido. ``RespostaCortada`` distingue respostas interrompidas para que o
orquestrador possa tentar novamente com parâmetros de amostragem diferentes.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from datetime import date, datetime, timedelta

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)
CURRENT_YEAR = date.today().year

_MESES_PT = {
    "janeiro": 1, "fevereiro": 2, "marco": 3, "março": 3, "abril": 4,
    "maio": 5, "junho": 6, "julho": 7, "agosto": 8, "setembro": 9,
    "outubro": 10, "novembro": 11, "dezembro": 12,
}

_DIAS_SEMANA_PT = {
    "segunda", "terca", "terça", "quarta", "quinta", "sexta",
    "sabado", "sábado", "domingo",
}


def _parse_validity_date(text: object) -> date | None:
    """Interpreta a validade nos formatos usuais encontrados em encartes.

    Aceita datas ISO, datas brasileiras com ou sem ano e mês por extenso.
    Textos que indiquem somente um dia da semana ou datas inválidas não são
    tratados como uma data absoluta.

    Args:
        text: Valor bruto devolvido pelo modelo no campo ``validade``.

    Returns:
        Data reconhecida ou ``None`` quando o valor não é uma data válida.
    """
    if not text:
        return None
    if isinstance(text, datetime):
        return text.date()
    if isinstance(text, date):
        return text
    if not isinstance(text, str):
        return None

    normalized_text = unicodedata.normalize("NFKD", text.casefold())
    normalized_text = "".join(
        character for character in normalized_text
        if not unicodedata.combining(character)
    ).strip()
    if not normalized_text:
        return None

    normalized_text = re.sub(
        r"^(validade|valido ate|valido|v[aá]lido at[eé]|ate)\s*[:\-]?\s*",
        "", normalized_text, flags=re.IGNORECASE,
    )

    iso_date_match = re.search(r"\b(\d{4})-(\d{2})-(\d{2})\b", normalized_text)
    if iso_date_match:
        try:
            return date(int(iso_date_match.group(1)), int(iso_date_match.group(2)), int(iso_date_match.group(3)))
        except ValueError:
            pass

    numeric_date_match = re.search(
        r"\b(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{2,4})\b",
        normalized_text,
    )
    if numeric_date_match:
        day, month, year = map(int, numeric_date_match.groups())
        if year < 100:
            year += 2000
        try:
            return date(year, month, day)
        except ValueError:
            pass

    textual_date_match = re.search(
        r"\b(\d{1,2})\s*(?:de\s+)?([a-z]+)(?:\s+de\s+(\d{4}))?",
        normalized_text,
    )
    if textual_date_match and textual_date_match.group(2) in _MESES_PT:
        day = int(textual_date_match.group(1))
        month = _MESES_PT[textual_date_match.group(2)]
        year = int(textual_date_match.group(3)) if textual_date_match.group(3) else CURRENT_YEAR
        try:
            return date(year, month, day)
        except ValueError:
            pass

    date_without_year_match = re.search(r"\b(\d{1,2})[/\-.](\d{1,2})\b", normalized_text)
    if date_without_year_match:
        day, month = map(int, date_without_year_match.groups())
        try:
            data = date(CURRENT_YEAR, month, day)
            if data < date.today() - timedelta(days=30):
                data = date(CURRENT_YEAR + 1, month, day)
            return data
        except ValueError:
            pass

    if any(day in normalized_text for day in _DIAS_SEMANA_PT):
        return None
    return None


def _normalize_validity(text: object) -> str | None:
    """Converte uma validade reconhecida para ISO ``YYYY-MM-DD``.

    Args:
        text: Valor bruto do campo de validade.

    Returns:
        Data em formato ISO ou ``None`` se não houver data reconhecível.
    """
    data = _parse_validity_date(text)
    if data is None:
        if text:
            logger.warning("Validade não reconhecida: %r", text)
        return None
    return data.isoformat()


def _clean_text(value: object) -> object:
    """Converte textos vazios ou nulos do modelo em ``None``.

    Args:
        value: Valor recebido de uma resposta estruturada.

    Returns:
        Texto aparado, valor original não textual ou ``None`` para marcadores
        vazios.
    """
    if not isinstance(value, str):
        return value
    value = value.strip()
    if value.lower() in {"", "null", "none", "n/a", "na", "-"}:
        return None
    return value


def _parse_float(value: object) -> float | None:
    """Converte números e preços brasileiros em ``float``.

    Args:
        value: Número ou texto como ``"R$ 12,90"``.

    Returns:
        Valor numérico convertido ou ``None`` para entradas inválidas.
    """
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = (
            value.strip().replace("R$", "").replace("r$", "")
            .replace(" ", "").replace(",", ".")
        )
        try:
            return float(text)
        except ValueError:
            return None
    return None


def _normalize_unit_of_measure(value: object) -> str | None:
    """Mapeia abreviações e nomes de unidade para os valores do esquema.

    Args:
        value: Unidade identificada pelo modelo, como ``"quilo"`` ou ``"L"``.

    Returns:
        ``g``, ``kg``, ``mg``, ``ml``, ``l``, ``un``, ``outro`` ou ``None``.
    """
    if value is None:
        return None
    if not isinstance(value, str):
        return "outro"
    unit = value.strip().lower().rstrip(".")
    if not unit:
        return None
    if unit in {"g", "gr", "grama", "gramas"}:
        return "g"
    if unit in {"kg", "quilo", "quilos", "quilograma", "quilogramas"}:
        return "kg"
    if unit in {"mg", "miligrama", "miligramas"}:
        return "mg"
    if unit in {"ml", "mililitro", "mililitros"}:
        return "ml"
    if unit in {"l", "lt", "litro", "litros"}:
        return "l"
    if unit in {"un", "und", "unid", "unidade", "unidades"}:
        return "un"
    return "outro"


class RespostaCortada(ValueError):
    """Indica que o modelo interrompeu a resposta por limite de tokens."""


class Variante(BaseModel):
    """Representa um produto/embalagem dentro de uma oferta promocional."""

    produto: str | None = Field(description="Nome e variante do produto.")
    marca: str | None = Field(description="Marca; null se ausente.")
    quantidade: float | None = Field(description="Valor numérico da embalagem.")
    unidade_medida: str | None = Field(
        description="Unidade: g|kg|mg|ml|l|un|outro."
    )

    @field_validator("produto", "marca", "unidade_medida", mode="before")
    @classmethod
    def _clean(cls, value: object) -> object:
        """Normaliza campos de texto vazios antes da validação."""
        return _clean_text(value)

    @field_validator("quantidade", mode="before")
    @classmethod
    def _parse_quantity(cls, value: object) -> float | None:
        """Converte a quantidade informada pelo modelo para número."""
        return _parse_float(value)

    @field_validator("unidade_medida", mode="after")
    @classmethod
    def _normalize_unit(cls, value: str | None) -> str | None:
        """Normaliza a unidade para o vocabulário aceito pelo esquema."""
        return _normalize_unit_of_measure(value)


class Oferta(BaseModel):
    """Representa uma oferta, que pode conter várias variantes pelo mesmo preço."""

    variantes: list[Variante]
    preco_atual: float | None = Field(
        description="Preço principal em reais, decimal."
    )
    preco_anterior: float | None = Field(description="Preço anterior explícito.")
    preco_condicao: float | None = Field(
        description=(
            "Preço alternativo COM CONDIÇÃO (ex.: preço do Clube, do App, "
            "do 'leve 3 pague 2'). É o segundo preço, MENOR que preco_atual, "
            "impresso logo abaixo/ao lado com a etiqueta da condição. "
            "null se a oferta tem só um preço."
        )
    )
    condicao: str | None = Field(
        description=(
            "Texto EXATO da condição como aparece na imagem "
            "(ex.: 'Clube Mais', 'App', 'leve 3 pague 2', 'Só com o cartão'). "
            "Preencha SEMPRE que houver preco_condicao; null caso contrário."
        )
    )

    @field_validator("condicao", mode="before")
    @classmethod
    def _clean(cls, value: object) -> object:
        """Normaliza a condição vazia antes da validação."""
        return _clean_text(value)

    @field_validator(
        "preco_atual", "preco_anterior", "preco_condicao", mode="before"
    )
    @classmethod
    def _parse_prices(cls, value: object) -> float | None:
        """Converte os preços da resposta para números decimais."""
        return _parse_float(value)


class ResultadoExtracao(BaseModel):
    """Resposta validada de classificação e extração de uma imagem."""

    encarte_valido: bool = Field(
        description="True somente se a imagem for encarte de supermercado."
    )
    validade: str | None = Field(default=None)
    ofertas: list[Oferta] = Field(default_factory=list)

    @field_validator("validade", mode="before")
    @classmethod
    def _clean(cls, value: object) -> object:
        """Remove marcadores vazios do campo de validade."""
        return _clean_text(value)

    @field_validator("validade", mode="after")
    @classmethod
    def _normalize(cls, value: str | None) -> str | None:
        """Normaliza datas reconhecíveis para ISO."""
        return _normalize_validity(value)


class ProdutosEncontrados(BaseModel):
    """Resposta da chamada que confirma produtos da lista de referência."""

    produtos: list[str] = Field(
        default_factory=list,
        description=(
            "Produtos da lista de referência que aparecem na imagem, "
            "escritos exatamente como na lista de referência."
        ),
    )