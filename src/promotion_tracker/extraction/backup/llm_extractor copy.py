"""
Extrator de ofertas de encartes de supermercado via Qwen2.5-VL / Ollama.

Esta versão é otimizada para execução em CPU (que é o cenário atual,
confirmado por `ollama ps` mostrando 100% CPU):

    - num_thread = núcleos físicos (maior ganho em CPU).
    - num_batch = 512 (prefill mais rápido).
    - NUM_CTX reduzido para 6144 (menos KV-cache, menos banda de RAM).
    - MAX_LADO reduzido para 1024 (menos tokens de visão no prefill).
    - Prompt e schema Pydantic enxutos.
    - Log de tokens/s de prefill e de geração por imagem, para diagnóstico.

Fluxo por imagem:
    1. O OCR (PaddleOCR) lê o texto e as POSIÇÕES de cada linha.
    2. Se nenhum produto do segmento (config/segments.yaml) aparece, a
       imagem é ignorada (sem chamar o LLM).
    3. Se aparece, o encarte é RECORTADO: só a região de cada produto
       encontrado (nome + preço + condição) segue para o LLM. O resto é
       descartado. Isso deixa o modelo mais rápido e mais preciso, pois cada
       recorte é lido em resolução bem maior que o encarte inteiro.
    4. O modelo classifica (encarte_valido) e extrai na mesma resposta.
    5. Uma segunda chamada ao LLM revisita a imagem inteira para casar os
       produtos do segments.yaml com o que aparece nela e preencher o campo
       `produtos_encontrados` do JSON.

Casamento de produtos (tolerante a erros de OCR):
    1. palavra inteira ("arroz" em "arroz tio joao");
    2. texto colado, sem espaços ("ARROZ5KG", "ARROZTIOJOAO");
    3. fuzzy via rapidfuzz, se instalado ("ARR0Z" ~ "arroz").

Encerramento do Ollama:
    - Inicia "ollama serve" se necessário.
    - Encerra o servidor em Ctrl+C, SIGTERM, SIGHUP, fechamento do terminal
      no Windows e via atexit, liberando a RAM.

Ordem de execução das funções:
    __init__() -> process_bronze_folder() -> _process_bronze_folder() ->
    para cada imagem: _find_segment_products() -> _extract_data() ->
    _query_model() -> _revisit_segment_products() -> _save_json()
"""

import atexit
import io
import json
import logging
import os
import re
import signal
import statistics
import subprocess
import time
import unicodedata
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import ollama
import numpy as np
import yaml
from PIL import Image, ImageDraw
from pydantic import BaseModel, Field, ValidationError, field_validator

try:
    from rapidfuzz import fuzz  # type: ignore
except Exception:  # rapidfuzz é opcional
    fuzz = None

# ----------------------------------------------------------------------
# Variáveis de ambiente recomendadas (definidas antes do ollama importar)
# ----------------------------------------------------------------------
# os.environ.setdefault("OLLAMA_FLASH_ATTENTION", "1")
# os.environ.setdefault("OLLAMA_KV_CACHE_TYPE", "q8_0")


logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Detecção de threads físicas
# ----------------------------------------------------------------------

def _detect_physical_cores() -> int:
    """
    Retorna o número de núcleos FÍSICOS (não lógicos). É o valor ideal
    para num_thread em llama.cpp em CPU: usar núcleos lógicos costuma
    piorar o desempenho por contenção de memória.

    Returns:
        int: Número de núcleos físicos (mínimo 1)
    """
    try:
        import psutil  # type: ignore
        physical_core_count = psutil.cpu_count(logical=False)
        if physical_core_count and physical_core_count > 0:
            return physical_core_count
    except Exception:
        pass

    logical_core_count = os.cpu_count() or 2
    return max(1, logical_core_count // 2)


# ----------------------------------------------------------------------
# Configurações do modelo
# ----------------------------------------------------------------------

# Encarte inteiro (~1300 tokens de visão) + prompt/schema (~1900) + resposta
# (até 4096) precisam caber na janela.
NUM_CTX = 8192
NUM_PREDICT = 4096
MAX_LADO = 1024
MIN_LADO_ENVIO = 672

NUM_THREAD = _detect_physical_cores()
NUM_BATCH = 512


# ----------------------------------------------------------------------
# Casamento de produtos (tolerância a erros de OCR)
# ----------------------------------------------------------------------

# Aceita o produto "colado" (sem espaços) dentro do texto também colado.
MATCH_JOINED_TEXT = True

# Usa rapidfuzz (se instalado) para tolerar letras trocadas pelo OCR.
MATCH_FUZZY = True
FUZZY_THRESHOLD = 90          # 0-100; suba se aparecerem falsos positivos
MIN_TOLERANT_CHARS = 5    # só aplica colado/fuzzy a termos com >= N letras

# Filtro anti "não é encarte": um encarte tem vários preços no texto do OCR.
# Se o OCR achar menos preços que isso, a imagem é ignorada sem chamar o LLM.
# Use 0 para desligar (ex.: se seus encartes separam "9" e "99" em linhas).
# Padrão 0 (desligado): a decisão fica com o LLM + validações pós-resposta,
# para nunca descartar um encarte real por falha de leitura do preço.
OCR_MIN_PRICES = 0

# OCR em duas passadas. 1) imagem inteira. 2) se nenhum produto do segmento
# foi achado, relê a imagem TODA em faixas sobrepostas e ampliadas (pega
# texto pequeno que a passada 1 perdeu). Custa mais tempo nas imagens sem
# produto; ponha False para desligar.
OCR_USE_STRIPS = True
OCR_STRIP_HEIGHT = 960        # px de altura de cada faixa (imagem original)
OCR_STRIP_OVERLAP = 0.25  # fração de sobreposição entre faixas
OCR_STRIP_SCALE = 1.5         # ampliação antes de ler cada faixa


# ----------------------------------------------------------------------
# Recorte da região dos produtos (usa as posições do OCR)
# ----------------------------------------------------------------------

# False -> envia o ENCARTE INTEIRO ao LLM (padrão). True -> só recortes.
CROP_PRODUCTS = False

CROP_X_PADDING = 0.02
CROP_Y_PADDING = 0.012
CROP_ALIGNMENT_TOLERANCE = 0.07
CROP_MAX_PRICE_DISTANCE = 0.30
CROP_MAX_HEIGHT_AFTER_PRICE = 0.35
CROP_FIRST_LINE_GAP = 3.5
CROP_NEXT_LINE_GAP = 1.3
CROP_BELOW_PRICE_GAP = 1.0
CROP_HEIGHT_RATIO = 1.5
CROP_MIN_MERGE_OVERLAP = 0.15
CROP_FALLBACK_HEIGHT = 0.155
CROP_FALLBACK_WIDTH = 0.15

COMPOSITE_MAX_HEIGHT = MAX_LADO
COMPOSITE_GAP = 24


# ----------------------------------------------------------------------
# Gerenciamento do servidor Ollama
# ----------------------------------------------------------------------

MANAGE_OLLAMA_SERVER = True
TERMINATE_EXISTING_SERVER = True
OLLAMA_START_TIMEOUT = 25


# ----------------------------------------------------------------------
# Validade: parsing e normalização
# ----------------------------------------------------------------------

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


def _parse_validity_date(text) -> date | None:
    """
    Converte o texto devolvido pelo LLM em um objeto `date`

    Tolera os formatos comuns de encarte:
        - ISO (2026-10-25)
        - BR (25/10/2026)
        - dia/mês sem ano (25/10) -> usa ano corrente
        - mês por extenso (25 de outubro de 2026 / 25 de outubro)
        - dia da semana isolado ("terça-feira") -> None

    Args:
        text: Texto devolvido pelo modelo na chave `validade`

    Returns:
        date | None: Data interpretada ou None se não for reconhecível
    """
    if not text:
        return None
    if isinstance(text, date) and not isinstance(text, datetime):
        return text
    if isinstance(text, datetime):
        return text.date()
    if not isinstance(text, str):
        return None

    t = unicodedata.normalize("NFKD", text.casefold())
    t = "".join(c for c in t if not unicodedata.combining(c))
    t = t.strip()
    if not t:
        return None

    # Remove prefixos comuns ("válido até", "validade:", etc.)
    t = re.sub(
        r"^(validade|valido ate|valido|v[aá]lido at[eé]|ate)\s*[:\-]?\s*",
        "", t, flags=re.IGNORECASE,
    )

    # ISO: 2026-10-25
    m = re.search(r"\b(\d{4})-(\d{2})-(\d{2})\b", t)
    if m:
        try:
            return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            pass

    # BR com ano: 25/10/2026 ou 25-10-2026
    m = re.search(r"\b(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{2,4})\b", t)
    if m:
        d, mo, a = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if a < 100:
            a += 2000
        try:
            return date(a, mo, d)
        except ValueError:
            pass

    # Mês por extenso: "25 de outubro de 2026" ou "25 de outubro"
    m = re.search(r"\b(\d{1,2})\s*(?:de\s+)?([a-z]+)(?:\s+de\s+(\d{4}))?", t)
    if m and m.group(2) in _MESES_PT:
        d = int(m.group(1))
        mo = _MESES_PT[m.group(2)]
        a = int(m.group(3)) if m.group(3) else CURRENT_YEAR
        try:
            return date(a, mo, d)
        except ValueError:
            pass

    # Dia/mês sem ano: 25/10
    m = re.search(r"\b(\d{1,2})[/\-.](\d{1,2})\b", t)
    if m:
        d, mo = int(m.group(1)), int(m.group(2))
        try:
            data = date(CURRENT_YEAR, mo, d)
            # Se já passou há mais de 30 dias, provavelmente é do ano seguinte
            if data < date.today() - timedelta(days=30):
                data = date(CURRENT_YEAR + 1, mo, d)
            return data
        except ValueError:
            pass

    # Dia da semana isolado -> não é data absoluta
    if any(day in t for day in _DIAS_SEMANA_PT):
        return None

    return None


def _normalize_validity(text) -> str | None:
    """
    Converte a saída do modelo em string ISO (YYYY-MM-DD) ou None

    Se o texto não puder ser interpretado como data, retorna None para que
    o JSON final não fique com valores como "terça-feira" em `validade`

    Args:
        text: Saída bruta do modelo no campo `validade`

    Returns:
        str | None: Data em formato ISO ou None se não reconhecida
    """
    data = _parse_validity_date(text)
    if data is None:
        if text:
            logger.warning("Validade não reconhecida: %r", text)
        return None
    return data.isoformat()


# ----------------------------------------------------------------------
# Funções auxiliares
# ----------------------------------------------------------------------

def _clean_text(value):
    """
    Normaliza um campo de texto vindo do modelo

    Remove espaços em volta e converte valores vazios/nulos textuais em None

    Args:
        value: Valor bruto (esperado string)

    Returns:
        str | None: Texto limpo ou None
    """
    if not isinstance(value, str):
        return value
    value = value.strip()
    if value.lower() in {"", "null", "none", "n/a", "na", "-"}:
        return None
    return value


def _parse_float(value):
    """
    Converte um valor de preço vindo do modelo em float

    Tolera string com "R$", espaços, vírgula decimal e valores numéricos

    Args:
        value: Valor bruto (string ou número)

    Returns:
        float | None: Preço convertido ou None
    """
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = (
            value.strip()
            .replace("R$", "").replace("r$", "")
            .replace(" ", "").replace(",", ".")
        )
        try:
            return float(text)
        except ValueError:
            return None
    return None


def _normalize_unit_of_measure(value):
    """
    Normaliza a unidade de medida para um conjunto fixo de valores

    Args:
        value: Unidade bruta ("kg", "quilo", "L", etc.)

    Returns:
        str | None: "g", "kg", "mg", "ml", "l", "un", "outro" ou None
    """
    if value is None:
        return None
    if not isinstance(value, str):
        return "outro"
    v = value.strip().lower().rstrip(".")
    if not v:
        return None
    if v in {"g", "gr", "grama", "gramas"}:
        return "g"
    if v in {"kg", "quilo", "quilos", "quilograma", "quilogramas"}:
        return "kg"
    if v in {"mg", "miligrama", "miligramas"}:
        return "mg"
    if v in {"ml", "mililitro", "mililitros"}:
        return "ml"
    if v in {"l", "lt", "litro", "litros"}:
        return "l"
    if v in {"un", "und", "unid", "unidade", "unidades"}:
        return "un"
    return "outro"


def normalize_search_text(text: str) -> str:
    """
    Normaliza um texto para comparação tolerante (case/acento-insensível)

    Minúsculas, sem acentos e só letras/números separados por espaço

    Args:
        text: Texto original

    Returns:
        str: Texto normalizado
    """
    text = unicodedata.normalize("NFKD", text.casefold())
    text = "".join(
        character for character in text
        if not unicodedata.combining(character)
    )
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text).split())


def _configure_logging(debug: bool) -> None:
    """
    Configura o logging da aplicação

    Args:
        debug: Se True, eleva o nível para DEBUG

    Returns:
        None
    """
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)-7s | %(message)s",
            datefmt="%H:%M:%S",
        )
    logger.setLevel(logging.DEBUG if debug else logging.INFO)
    for name in ("httpx", "httpcore", "PIL"):
        logging.getLogger(name).setLevel(logging.WARNING)


def product_matches(
    product: str, normalized_text: str, allow_unordered_words: bool = False
) -> bool:
    """
    Verifica se um produto do segmento aparece em um texto normalizado

    Tenta, nesta ordem: palavra inteira, palavras soltas (opcional), texto
    colado (sem espaços) e fuzzy. O parâmetro `palavras_soltas` é útil para
    encartes em colunas, onde o OCR embaralha a ordem das linhas

    Args:
        product: Produto do segmento (já normalizado)
        normalized_text: Texto do OCR (já normalizado)
        allow_unordered_words: Se True, aceita o produto com as palavras separadas

    Returns:
        bool: True se o produto casar com o texto
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


# ----------------------------------------------------------------------
# Esquemas Pydantic
# ----------------------------------------------------------------------

class RespostaCortada(ValueError):
    """O modelo atingiu o limite de tokens (geralmente loop de repetição)."""


class Variante(BaseModel):
    """Variante de um produto dentro de uma oferta."""

    produto: str | None = Field(description="Nome e variante do produto.")
    marca: str | None = Field(description="Marca; null se ausente.")
    quantidade: float | None = Field(description="Valor numérico da embalagem.")
    unidade_medida: str | None = Field(
        description="Unidade: g|kg|mg|ml|l|un|outro."
    )

    @field_validator("produto", "marca", "unidade_medida", mode="before")
    @classmethod
    def _clean(cls, value):
        """Limpa campos de texto vazios/nulos."""
        return _clean_text(value)

    @field_validator("quantidade", mode="before")
    @classmethod
    def _parse_quantity(cls, value):
        """Converte a quantidade em float."""
        return _parse_float(value)

    @field_validator("unidade_medida", mode="after")
    @classmethod
    def _normalize_unit(cls, value):
        """Normaliza a unidade de medida."""
        return _normalize_unit_of_measure(value)


class Oferta(BaseModel):
    """Oferta de um ou mais produtos com um preço principal."""

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
    def _clean(cls, value):
        """Limpa o texto da condição."""
        return _clean_text(value)

    @field_validator(
        "preco_atual", "preco_anterior", "preco_condicao",
        mode="before",
    )
    @classmethod
    def _parse_prices(cls, value):
        """Converte os preços em float."""
        return _parse_float(value)


class ResultadoExtracao(BaseModel):
    """Resposta do modelo para uma chamada de extração."""

    encarte_valido: bool = Field(
        description="True somente se a imagem for encarte de supermercado."
    )
    validade: str | None = Field(default=None)
    ofertas: list[Oferta] = Field(default_factory=list)

    @field_validator("validade", mode="before")
    @classmethod
    def _clean(cls, value):
        """Limpa o campo de validade."""
        return _clean_text(value)

    @field_validator("validade", mode="after")
    @classmethod
    def _normalize(cls, value):
        """Normaliza a validade para ISO (YYYY-MM-DD) ou None."""
        return _normalize_validity(value)


class ProdutosEncontrados(BaseModel):
    """Resposta do modelo na revisita de produtos do segments.yaml."""

    produtos: list[str] = Field(
        default_factory=list,
        description=(
            "Produtos da lista de referência que aparecem na imagem, "
            "escritos exatamente como na lista de referência."
        ),
    )


# ----------------------------------------------------------------------
# Prompt enxuto (~750 tokens)
# ----------------------------------------------------------------------

PROMPT = f"""
Você analisa imagens de encartes de supermercados brasileiros.

CLASSIFICAÇÃO (seja rigoroso):
- encarte_valido=true SOMENTE se a imagem mostrar, em formato de material
  promocional de supermercado, produtos com PREÇO em reais (R$ ou
  números grandes de preço) ao lado de cada produto. Um recorte de
  encarte também vale, desde que tenha produto E preço visíveis.
- encarte_valido=false para: foto de produto, embalagem ou prateleira sem
  etiqueta de preço; foto pessoal, paisagem, comida, receita; logo,
  banner institucional, anúncio de marca, cupom, print de rede social,
  meme, texto solto; imagem sem nenhum preço legível.
- Na dúvida, ou se não houver preço legível: encarte_valido=false.
- Se false: ofertas=[] e validade=null. Não invente.

RECORTES:
- A imagem pode ser um recorte do encarte, ou vários recortes empilhados
  na vertical (separados por uma linha cinza). Extraia só ofertas
  COMPLETAS (nome e preço visíveis); ignore itens cortados na borda.

SE ENCARTE, extraia TODAS as ofertas.

VALIDADE (campo "validade"):
- Procure na imagem por textos como "Válido até", "Validade:",
  "Ofertas válidas de X a Y", "Promoção válida até".
- Devolva SEMPRE no formato ISO "YYYY-MM-DD" (ex.: "2026-10-25").
- Se a imagem trouxer só dia/mês ("até 25/10") use o ano corrente
  ({CURRENT_YEAR}); se essa data já passou há mais de 30 dias, use
  {CURRENT_YEAR + 1}.
- Se a imagem trouxer mês por extenso ("25 de outubro de 2026"), converta
  para ISO.
- Se a imagem trouxer APENAS um dia da semana ("terça-feira") sem data,
  devolva null (não é data absoluta).
- Se não houver validade na imagem, devolva null.

VARIANTE (por produto):
- produto: nome e variante como escrito.
- marca: identificável ou null.
- quantidade: número sem unidade ("390g"→390, "1,5L"→1.5).
- unidade_medida: "g"|"kg"|"mg"|"ml"|"l"|"un"|"outro"|null.

OFERTA (por preço):
- preco_atual: preço principal em reais (decimal). Sempre presente.
- preco_anterior: preço anterior explícito ou null.
- preco_condicao: SEGUNDO preço da oferta, quando aparece com uma
  CONDIÇÃO (Clube, App, cartão, "leve 3 pague 2", "só para membros").
  É o preço menor, impresso logo abaixo/ao lado do principal, com uma
  etiqueta colorida ou em negrito da condição. Ex. na tela:
      "R$ 25,90"
      "R$ 22,90 no Clube Mais"
  → preco_atual=25.90, preco_condicao=22.90, condicao="Clube Mais".
  Se houver SÓ um preço, preco_condicao=null e condicao=null.
- condicao: texto EXATO da etiqueta da condição, como está escrito
  ("Clube Mais", "App", "Cartão Fidelidade", "leve 3 pague 2"). SEMPRE
  preencha se houver preco_condicao. SEMPRE deixe null se não houver
  segundo preço com condição.

VÁRIOS PRODUTOS MESMO PREÇO:
Produtos ligados por "OU", "/", vírgula ou "e" sob um único preço →
uma oferta com várias variantes. Ex.: "IOGURTE NESTLÉ MORANGO OU COCO
170G R$2,49" → 2 variantes (quantidade=170, unidade_medida="g"),
preco_atual=2.49. Vários sabores não nomeados → variante "Sabores".

EXEMPLOS:
1) "ARROZ TIO JOÃO 5KG — R$25,90 / R$22,90 Clube Mais"
   variante: {{produto:"Arroz Tio João", marca:"Tio João",
              quantidade:5, unidade_medida:"kg"}}
   oferta:   {{preco_atual:25.90, preco_condicao:22.90,
              condicao:"Clube Mais"}}

2) "CAFÉ PILÃO 500G — R$18,90"  (só um preço)
   variante: {{produto:"Café Pilão", quantidade:500, unidade_medida:"g"}}
   oferta:   {{preco_atual:18.90, preco_anterior:null,
              preco_condicao:null, condicao:null}}

REGRAS:
1. Não invente dados. Associe cada preço ao produto visualmente correto.
2. Vírgula é decimal: "R$12,99"→12.99; "1,5L"→1.5.
3. Não confunda peso/volume/código/parcelas com preço.
4. "REF:1234" é código, não preço. Ignore "CADA" junto ao preço.
5. Não descarte oferta por campo secundário ilegível.
6. Não duplique ofertas.
7. `condicao` só existe junto de `preco_condicao`. Se um dos dois está
   presente, o outro TAMBÉM deve estar.
"""

PROMPT_PRODUCTS = """
Você recebe uma imagem de encarte de supermercado e uma LISTA DE PRODUTOS
DE REFERÊNCIA. Sua única tarefa é dizer quais itens da lista aparecem na
imagem, mesmo que escritos com variações (marca, tamanho, sabor, plural,
abreviação).

LISTA DE REFERÊNCIA:
{lista}

REGRAS:
1. Devolva APENAS produtos da lista acima, copiados LITERALMENTE como estão
   na lista (não invente, não traduza, não reescreva).
2. Inclua um item se QUALQUER produto visível na imagem for da mesma
   categoria daquele item (ex.: "arroz" na lista casa com "Arroz Tio João
   5kg" na imagem).
3. Se nenhum item da lista aparece, devolva {{"produtos": []}}.
4. Não inclua itens duplicados.
"""


# ----------------------------------------------------------------------
# Geometria do OCR: localiza a região (bloco) de cada oferta
# ----------------------------------------------------------------------

_RE_PRICE = re.compile(r"(?<!\d)\d{1,4}\s*[.,]\s*\d{2}(?!\d)")
_RE_DIGITS_ONLY = re.compile(r"^\d{1,4}$")
_RE_CENTS = re.compile(r"^[.,]\s*\d{2}$")


def _looks_like_price(text: str) -> bool:
    """
    Heurística tolerante para o filtro "tem preço?"

    Aceita "12,90", "R$", números soltos de 1-4 dígitos ("12") e centavos
    soltos (",90"), pois o OCR costuma dividir o preço em vários pedaços

    Args:
        text: Texto de uma linha do OCR

    Returns:
        bool: True se o texto parece conter um preço
    """
    t = text.strip()
    if not t:
        return False
    if _RE_PRICE.search(t):
        return True
    if "r$" in t.lower().replace(" ", ""):
        return True
    return bool(_RE_DIGITS_ONLY.match(t) or _RE_CENTS.match(t))


@dataclass
class LinhaOCR:
    """Uma linha de texto detectada pelo OCR, com sua caixa na imagem."""

    text: str
    normalized_text: str
    x0: float
    y0: float
    x1: float
    y_end: float

    @property
    def width(self) -> float:
        """Largura da caixa."""
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        """Altura da caixa."""
        return self.y_end - self.y0

    @property
    def center_y(self) -> float:
        """Centro vertical da caixa."""
        return (self.y0 + self.y_end) / 2


@dataclass
class RegiaoOferta:
    """Região do encarte (x0, y0, x1, y1) que contém a oferta de um produto."""

    box: tuple[int, int, int, int]
    products: list[str]
    method: str  # "ancora_preco" ou "janela_fixa"
    center: tuple[float, float] | None = None


@dataclass
class AnaliseOCR:
    """Resultado da leitura OCR de uma imagem."""

    found_products: list[str]
    regions: list[RegiaoOferta]
    image_size: tuple[int, int]
    use_full_image: bool
    reason: str = ""


class LocalizadorOfertas:
    """
    A partir das linhas do OCR, descobre o bloco de cada oferta

    Ideia: toda oferta tem um PREÇO. O bloco vai do início do nome do produto
    (acima do preço) até o início do nome da próxima oferta na mesma coluna,
    e na horizontal até a coluna vizinha. Os textos de botões/selos abaixo do
    preço (ex.: "Ative a oferta no Clube") ficam dentro do bloco, porque são
    a CONDIÇÃO do preço

    Se não achar o preço, usa uma janela fixa ao redor do nome
    """

    def __init__(self, lines: list[LinhaOCR], width: int, height: int):
        """
        Pré-calcula as estatísticas geométricas usadas na localização

        Args:
            lines: Linhas do OCR com posição
            width: Largura da imagem em pixels
            height: Altura da imagem em pixels
        """
        self.lines = lines
        self.width = float(width)
        self.height = float(height)

        line_heights = [l.height for l in lines if l.height > 0] or [20.0]
        lh_geral = statistics.median(line_heights)

        self.precos = [
            i for i, l in enumerate(lines) if self._is_price(l, lh_geral)
        ]
        self._precos_set = set(self.precos)

        text_line_heights = [
            l.height for i, l in enumerate(lines)
            if i not in self._precos_set and l.height > 0
        ]
        self.lh = statistics.median(text_line_heights) if text_line_heights else lh_geral
        self.tol_x = CROP_ALIGNMENT_TOLERANCE * self.width
        self._pares = self._find_adjacent_pairs()

    @staticmethod
    def _is_price(line: LinhaOCR, lh_geral: float) -> bool:
        """
        Verifica se uma linha é um preço (não um texto qualquer)

        Args:
            line: Linha do OCR
            lh_geral: Altura típica de linha na imagem

        Returns:
            bool: True se a linha parece um preço
        """
        text = line.text.strip()
        if _RE_PRICE.search(text):
            return sum(c.isalpha() for c in text) <= 6
        return bool(_RE_DIGITS_ONLY.match(text)) and line.height >= 1.4 * lh_geral

    def _are_adjacent(self, a: LinhaOCR, b: LinhaOCR) -> bool:
        """
        Verifica se duas linhas são vizinhas (mesma coluna ou mesma linha)

        Args:
            a: Primeira linha
            b: Segunda linha

        Returns:
            bool: True se `b` vem logo abaixo ou logo ao lado de `a`
        """
        gap_v = b.y0 - a.y_end
        horizontal_overlap = min(a.x1, b.x1) - max(a.x0, b.x0)
        if (
            b.y0 > a.y0
            and -0.2 * self.lh <= gap_v <= self.lh
            and horizontal_overlap > 0.3 * min(a.width, b.width)
        ):
            return True

        vertical_overlap = min(a.y_end, b.y_end) - max(a.y0, b.y0)
        gap_h = b.x0 - a.x1
        return (
            vertical_overlap > 0.5 * min(a.height, b.height)
            and 0 <= gap_h <= 1.5 * self.lh
        )

    def _find_adjacent_pairs(self) -> list[tuple[int, int]]:
        """
        Calcula todos os pares de linhas vizinhas (para buscar produtos
        divididos em duas linhas)

        Returns:
            list[tuple[int, int]]: Lista de pares (i, j)
        """
        return [
            (i, j)
            for i, a in enumerate(self.lines)
            for j, b in enumerate(self.lines)
            if i != j and self._are_adjacent(a, b)
        ]

    def find_occurrences(
        self, products: list[str]
    ) -> list[tuple[str, tuple[int, ...]]]:
        """
        Acha em quais linhas cada produto do segmento aparece

        Considera o produto numa linha só ou dividido em duas linhas vizinhas

        Args:
            products: Lista de produtos do segmento (normalizados)

        Returns:
            list[tuple[str, tuple[int, ...]]]: Lista de (produto, índices)
        """
        occurrences: list[tuple[str, tuple[int, ...]]] = []

        for product in products:
            for i, line in enumerate(self.lines):
                if product_matches(product, line.normalized_text):
                    occurrences.append((product, (i,)))

            for i, j in self._pares:
                a, b = self.lines[i], self.lines[j]
                combined_text = f"{a.normalized_text} {b.normalized_text}"
                if (
                    product_matches(product, combined_text)
                    and not product_matches(product, a.normalized_text)
                    and not product_matches(product, b.normalized_text)
                ):
                    occurrences.append((product, (i, j)))

        return occurrences

    def _walk_lines(self, i0: int, direction: int, use_height_limit: bool = False) -> list[int]:
        """
        Coleta as linhas da mesma coluna, em sequência, a partir de uma linha

        Para quando a lacuna vertical fica grande demais. Com
        `usar_altura=True`, também para quando a altura da letra muda muito
        (separa o nome do texto dos botões)

        Args:
            i0: Índice da linha de referência
            direction: -1 (para cima) ou +1 (para baixo)
            use_height_limit: Se True, também interrompe ao mudar a altura da letra

        Returns:
            list[int]: Índices das linhas coletadas
        """
        ref = self.lines[i0]
        candidates = []

        for j, line in enumerate(self.lines):
            if j == i0 or j in self._precos_set:
                continue
            if abs(line.x0 - ref.x0) > self.tol_x:
                continue
            if direction < 0 and line.y_end <= ref.y0 + 0.3 * self.lh:
                candidates.append(j)
            elif direction > 0 and line.y0 >= ref.y_end - 0.3 * self.lh:
                candidates.append(j)

        if direction < 0:
            candidates.sort(key=lambda j: -self.lines[j].y_end)
        else:
            candidates.sort(key=lambda j: self.lines[j].y0)

        collected_indices: list[int] = []
        edge = ref.y0 if direction < 0 else ref.y_end
        reference_height = None

        for j in candidates:
            line = self.lines[j]
            gap = (edge - line.y_end) if direction < 0 else (line.y0 - edge)

            if not collected_indices:
                deadline = CROP_FIRST_LINE_GAP * self.lh
            elif direction < 0:
                deadline = CROP_NEXT_LINE_GAP * self.lh
            else:
                deadline = CROP_BELOW_PRICE_GAP * self.lh

            if gap > deadline:
                break

            if use_height_limit and reference_height is not None:
                ratio = line.height / reference_height if reference_height else 1.0
                if ratio > CROP_HEIGHT_RATIO or ratio < 1 / CROP_HEIGHT_RATIO:
                    break

            collected_indices.append(j)
            edge = line.y0 if direction < 0 else line.y_end
            if reference_height is None:
                reference_height = line.height

        return collected_indices

    def _find_associated_price(self, idxs: tuple[int, ...]) -> int | None:
        """
        Acha o preço da oferta: na própria linha ou o mais próximo abaixo

        Args:
            idxs: Índices das linhas onde o produto aparece

        Returns:
            int | None: Índice da linha do preço ou None se não achar
        """
        for i in idxs:
            if i in self._precos_set:
                return i

        base = [self.lines[i] for i in idxs]
        minimum_x0 = min(l.x0 for l in base)
        minimum_y0 = min(l.y0 for l in base)
        maximum_y1 = max(l.y_end for l in base)

        best_index, best_distance = None, None
        for p in self.precos:
            line = self.lines[p]
            if line.y0 < minimum_y0:
                continue
            if abs(line.x0 - minimum_x0) > 1.5 * self.tol_x:
                continue
            distance = line.y0 - maximum_y1
            if distance > CROP_MAX_PRICE_DISTANCE * self.height:
                continue
            if best_index is None or distance < best_distance:
                best_index, best_distance = p, distance

        return best_index

    def region_for(self, product: str, idxs: tuple[int, ...]) -> RegiaoOferta:
        """
        Calcula a região (caixa) da oferta de um produto

        Args:
            product: Produto do segmento
            idxs: Índices das linhas onde o produto aparece

        Returns:
            RegiaoOferta: Região com a caixa, produtos e método usado
        """
        width, height = self.width, self.height
        x_padding, y_padding = CROP_X_PADDING * width, CROP_Y_PADDING * height
        base = [self.lines[i] for i in idxs]

        price_index = self._find_associated_price(idxs)
        if price_index is None:
            return self._fixed_window(product, base)

        price = self.lines[price_index]
        name = [self.lines[j] for j in self._walk_lines(price_index, -1, use_height_limit=True)]
        block = base + name + [price]

        top = min(l.y0 for l in block) - y_padding
        left = min(l.x0 for l in block) - x_padding
        name_x1 = max(l.x1 for l in block)

        # Limite inferior: começo do nome da próxima oferta na coluna
        next_items = [
            p for p in self.precos
            if p != price_index
            and self.lines[p].y0 > price.y_end
            and abs(self.lines[p].x0 - price.x0) <= self.tol_x
        ]
        if next_items:
            p2 = min(next_items, key=lambda p: self.lines[p].y0)
            nome2 = [
                self.lines[j]
                for j in self._walk_lines(p2, -1, use_height_limit=True)
            ]
            topo2 = min([self.lines[p2].y0] + [l.y0 for l in nome2])
            bottom = topo2 - 0.004 * height
        else:
            below_lines = [self.lines[j] for j in self._walk_lines(price_index, +1)]
            bottom = max([price.y_end] + [l.y_end for l in below_lines]) + y_padding

        bottom = max(bottom, price.y_end + self.lh)
        bottom = min(bottom, price.y_end + CROP_MAX_HEIGHT_AFTER_PRICE * height, height)

        # Limite direito: começo da coluna vizinha (se existir)
        neighbors = [
            p for p in self.precos
            if p != price_index
            and self.lines[p].x0 - price.x0 > 0.15 * width
            and top <= self.lines[p].center_y <= bottom
        ]
        if neighbors:
            pv = min(neighbors, key=lambda p: self.lines[p].x0)
            x_nome_viz = [
                self.lines[j].x0
                for j in self._walk_lines(pv, -1, use_height_limit=True)
            ]
            right_limit = min([self.lines[pv].x0] + x_nome_viz) - 0.03 * width
        else:
            right_limit = width

        right = max(right_limit, name_x1 + x_padding, price.x1 + x_padding)

        x0, y0 = max(0.0, left), max(0.0, top)
        x1, y_end = min(width, right), min(height, bottom)

        if (x1 - x0) < 0.10 * width or (y_end - y0) < 2 * self.lh:
            return self._fixed_window(product, base)

        return RegiaoOferta(
            box=(int(x0), int(y0), int(x1), int(y_end)),
            products=[product],
            method="ancora_preco",
            center=self._calculate_center(base),
        )

    @staticmethod
    def _calculate_center(base: list[LinhaOCR]) -> tuple[float, float]:
        """
        Calcula o centro médio de um conjunto de linhas

        Args:
            base: Linhas de referência

        Returns:
            tuple[float, float]: Centro (x, y) em pixels
        """
        return (
            sum((l.x0 + l.x1) / 2 for l in base) / len(base),
            sum(l.center_y for l in base) / len(base),
        )

    def _fixed_window(self, product: str, base: list[LinhaOCR]) -> RegiaoOferta:
        """
        Gera uma janela fixa ao redor do nome (fallback sem preço)

        Args:
            product: Produto do segmento
            base: Linhas onde o produto aparece

        Returns:
            RegiaoOferta: Região com método "janela_fixa"
        """
        width, height = self.width, self.height
        x_padding, y_padding = CROP_X_PADDING * width, CROP_Y_PADDING * height

        x0 = max(0.0, min(l.x0 for l in base) - x_padding)
        y0 = max(0.0, min(l.y0 for l in base) - y_padding)
        x1 = min(width, max(l.x1 for l in base) + x_padding + CROP_FALLBACK_WIDTH * width)
        y_end = min(height, max(l.y_end for l in base) + CROP_FALLBACK_HEIGHT * height)

        return RegiaoOferta(
            box=(int(x0), int(y0), int(x1), int(y_end)),
            products=[product],
            method="janela_fixa",
            center=self._calculate_center(base),
        )


def merge_regions(regions: list[RegiaoOferta]) -> list[RegiaoOferta]:
    """
    Une regiões que se sobrepõem (ex.: dois produtos no mesmo bloco)

    Antes, descarta as "janelas fixas" cujo texto está DENTRO de um bloco já
    localizado pelo preço: é o caso do nome do produto impresso na embalagem
    da foto (ex.: "BATATA" no pacote), que não é uma oferta nova

    Args:
        regions: Lista de regiões detectadas

    Returns:
        list[RegiaoOferta]: Regiões mescladas e ordenadas
    """
    anchors = [r.box for r in regions if r.method == "ancora_preco"]

    def is_inside_block(region: RegiaoOferta) -> bool:
        """Verifica se o centro da região cai dentro de algum bloco-âncora."""
        if region.center is None:
            return False
        cx, center_y = region.center
        return any(x0 <= cx <= x1 and y0 <= center_y <= y_end for x0, y0, x1, y_end in anchors)

    pending_items = [
        r for r in regions
        if not (r.method == "janela_fixa" and is_inside_block(r))
    ]
    merged_regions: list[RegiaoOferta] = []

    while pending_items:
        current = pending_items.pop(0)
        changed = True
        while changed:
            changed = False
            for other_item in list(pending_items):
                ax0, ay0, ax1, ay1 = current.box
                bx0, by0, bx1, by1 = other_item.box
                inter_w = min(ax1, bx1) - max(ax0, bx0)
                inter_h = min(ay1, by1) - max(ay0, by0)
                menor_area = min((ax1 - ax0) * (ay1 - ay0), (bx1 - bx0) * (by1 - by0))
                if (
                    inter_w > 0 and inter_h > 0
                    and inter_w * inter_h >= CROP_MIN_MERGE_OVERLAP * menor_area
                ):
                    current = RegiaoOferta(
                        box=(min(ax0, bx0), min(ay0, by0), max(ax1, bx1), max(ay1, by1)),
                        products=sorted(set(current.products) | set(other_item.products)),
                        method=(
                            "janela_fixa"
                            if "janela_fixa" in (current.method, other_item.method)
                            else "ancora_preco"
                        ),
                    )
                    pending_items.remove(other_item)
                    changed = True
        merged_regions.append(current)

    merged_regions.sort(key=lambda r: (r.box[1], r.box[0]))
    return merged_regions


# ----------------------------------------------------------------------
# Extrator
# ----------------------------------------------------------------------

class LLMExtractor:
    """
    Processa imagens da camada Bronze com Qwen2.5-VL via Ollama

    O OCR acha os produtos do segmento e o encarte é recortado só nas
    regiões desses produtos. O modelo classifica e extrai na mesma resposta,
    validada com Pydantic. Depois, uma segunda chamada ao LLM revisita a
    imagem inteira para casar os produtos do segments.yaml com o que aparece
    nela e preencher o campo `produtos_encontrados`

    Encerra o servidor Ollama ao final (fim do lote, Ctrl+C, SIGTERM/SIGHUP,
    fechamento do terminal no Windows, atexit)

    Ordem de execução das funções:
        __init__() -> process_bronze_folder() -> _process_bronze_folder() ->
        para cada imagem: _find_segment_products() -> _extract_data() ->
        _query_model() -> _revisit_segment_products() -> _save_json()
    """

    IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}

    def __init__(
        self,
        model_name: str = "qwen2.5vl:latest",
        use_ocr_images: bool = False,
        skip_existing: bool = True,
        debug: bool = False,
        max_images: int | None = None,
    ) -> None:
        """
        Inicializa o extrator e prepara as pastas e o servidor Ollama

        Args:
            model_name: Nome do modelo no Ollama
            use_ocr_images: Se True, processa imagens terminadas em "_ocr"
            skip_existing: Se True, ignora imagens já processadas (JSON existe)
            debug: Se True, salva a resposta bruta e a imagem enviada
            max_images: Limite opcional de imagens por execução
        """
        _configure_logging(debug)

        self.base_dir = Path(__file__).resolve().parents[3]
        self.segments_path = self.base_dir / "config" / "segments.yaml"
        self.bronze_dir = self.base_dir / "data" / "bronze"
        self.silver_dir = self.base_dir / "data" / "silver"
        self.segment_products = self._load_segment_products()
        self._ocr_reader = None

        self.model_name = model_name
        self.use_ocr_images = use_ocr_images
        self.skip_existing = skip_existing
        self.debug = debug
        self.max_images = max_images

        self._stats = self._new_stats()
        self._modelo_utilizado = False
        self._ollama_process = None
        self._fechando = False
        self._fechado = False
        self._console_ctrl_handler = None
        self._motivo_ignorada = ""

        if not self.bronze_dir.exists():
            raise FileNotFoundError(
                f"Pasta Bronze não encontrada: {self.bronze_dir}"
            )
        self.silver_dir.mkdir(parents=True, exist_ok=True)

        self._register_exit_handlers()

        if MANAGE_OLLAMA_SERVER:
            self._start_server_if_needed()

        logger.info("Modelo: %s | debug: %s", self.model_name, self.debug)
        logger.info(
            "Extração: NUM_CTX=%d | NUM_PREDICT=%d | MAX_LADO=%d px",
            NUM_CTX, NUM_PREDICT, MAX_LADO,
        )
        logger.info(
            "CPU: NUM_THREAD=%d (físicos) | NUM_BATCH=%d",
            NUM_THREAD, NUM_BATCH,
        )
        logger.info(
            "Recorte por produto: %s",
            "ligado" if CROP_PRODUCTS else "desligado (encarte inteiro)",
        )
        logger.info(
            "Casamento: colado=%s | fuzzy=%s (limiar %d)",
            MATCH_JOINED_TEXT,
            "ligado" if (MATCH_FUZZY and fuzz is not None)
            else "desligado (instale rapidfuzz)" if MATCH_FUZZY
            else "desligado",
            FUZZY_THRESHOLD,
        )
        logger.info(
            "Produtos do segmento (%d): %s",
            len(self.segment_products), self.segment_products,
        )
        logger.info("Bronze: %s", self.bronze_dir)
        logger.info("Silver: %s", self.silver_dir)

    @staticmethod
    def _new_stats() -> dict:
        """
        Cria um dicionário de estatísticas zerado

        Returns:
            dict: Estatísticas de ofertas, produtos, compartilhadas e avisos
        """
        return {"ofertas": 0, "produtos": 0, "compartilhadas": 0, "avisos": 0}

    # ------------------------------------------------------------------
    # Servidor Ollama
    # ------------------------------------------------------------------

    def _start_server_if_needed(self) -> None:
        """
        Inicia o "ollama serve" caso ainda não esteja disponível

        Raises:
            RuntimeError: Se o executável não for encontrado ou o processo
                encerrar durante a inicialização
            TimeoutError: Se o servidor não ficar disponível no prazo
        """
        try:
            ollama.list()
            logger.info("Ollama já está disponível; usando o servidor existente.")
            return
        except Exception:
            pass

        creation_flags = 0
        if os.name == "nt":
            creation_flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)

        try:
            self._ollama_process = subprocess.Popen(
                ["ollama", "serve"],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creation_flags=creation_flags,
            )
        except FileNotFoundError:
            raise RuntimeError(
                "Executável 'ollama' não encontrado."
            ) from None

        deadline = time.monotonic() + OLLAMA_START_TIMEOUT
        while time.monotonic() < deadline:
            if self._ollama_process.poll() is not None:
                self._ollama_process = None
                raise RuntimeError(
                    "O processo 'ollama serve' encerrou durante a inicialização."
                )
            try:
                ollama.list()
                logger.info("Servidor Ollama iniciado por este programa.")
                return
            except Exception:
                time.sleep(0.4)

        self._stop_owned_server()
        raise TimeoutError("O servidor Ollama não ficou disponível.")

    def _stop_owned_server(self) -> None:
        """
        Encerra o servidor Ollama iniciado por esta instância

        Returns:
            None
        """
        processo = self._ollama_process
        self._ollama_process = None
        if processo is None or processo.poll() is not None:
            return
        logger.info("Encerrando o servidor Ollama iniciado por este programa...")
        try:
            processo.terminate()
            try:
                processo.wait(timeout=5)
            except subprocess.TimeoutExpired:
                processo.kill()
                try:
                    processo.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    logger.error("O processo Ollama não encerrou após kill().")
        except Exception:
            logger.exception("Falha ao encerrar o servidor Ollama iniciado.")

    def _stop_existing_server(self) -> None:
        """
        Encerra processos Ollama já em execução no sistema

        Returns:
            None
        """
        logger.warning(
            "Encerrando processos Ollama já em execução "
            "(ENCERRAR_SERVIDOR_EXISTENTE=True)..."
        )
        try:
            if os.name == "nt":
                for image in ("ollama app.exe", "ollama.exe"):
                    subprocess.run(
                        ["taskkill", "/F", "/IM", image],
                        capture_output=True, timeout=10,
                    )
            else:
                subprocess.run(
                    ["pkill", "-x", "ollama"],
                    capture_output=True, timeout=10,
                )
        except Exception:
            logger.exception("Falha ao encerrar servidor Ollama existente.")

    # ------------------------------------------------------------------
    # Handlers de saída
    # ------------------------------------------------------------------

    def _register_exit_handlers(self) -> None:
        """
        Registra handlers para encerrar o servidor em qualquer saída

        Returns:
            None
        """
        for sig_name in ("SIGINT", "SIGTERM", "SIGHUP", "SIGBREAK"):
            sig = getattr(signal, sig_name, None)
            if sig is None:
                continue
            try:
                signal.signal(sig, self._handle_signal)
            except (ValueError, OSError):
                pass
        try:
            atexit.register(self.close)
        except Exception:
            logger.debug("Falha ao registrar atexit.", exc_info=True)
        if os.name == "nt":
            self._register_windows_console_close()

    def _register_windows_console_close(self) -> None:
        """
        Registra o handler de fechamento do console no Windows

        Returns:
            None
        """
        try:
            import ctypes
            from ctypes import wintypes
            kernel32 = ctypes.windll.kernel32
            HandlerRoutine = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.DWORD)
            extractor_ref = self

            @HandlerRoutine
            def handler(event_type):
                """Libera os recursos quando o console é fechado."""
                if event_type in (2, 5, 6):
                    try:
                        extractor_ref.close()
                    except Exception:
                        pass
                    return True
                return False

            kernel32.SetConsoleCtrlHandler(handler, True)
            self._console_ctrl_handler = handler
        except Exception:
            logger.debug(
                "Falha ao registrar handler de console.", exc_info=True
            )

    def _handle_signal(self, signum, frame) -> None:
        """
        Trata sinais de encerramento (Ctrl+C, SIGTERM, etc.)

        Args:
            signum: Número do sinal recebido
            frame: Frame atual (não utilizado)

        Returns:
            None
        """
        if self._fechando or self._fechado:
            raise KeyboardInterrupt
        name = getattr(signal.Signals(signum), "name", str(signum))
        logger.warning("Sinal %s recebido. Liberando recursos...", name)
        try:
            self.close()
        except Exception:
            logger.exception("Erro ao liberar recursos no handler de sinal.")
        if signum == getattr(signal, "SIGINT", None):
            raise KeyboardInterrupt
        raise SystemExit(128 + signum)

    # ------------------------------------------------------------------
    # Descoberta
    # ------------------------------------------------------------------

    def _json_path(self, image_path: Path) -> Path:
        """
        Calcula o caminho do JSON de saída correspondente a uma imagem

        Args:
            image_path: Caminho da imagem na Bronze

        Returns:
            Path: Caminho do JSON na Silver
        """
        relative_path = image_path.relative_to(self.bronze_dir)
        name = f"{image_path.stem.removesuffix('_ocr')}.json"
        return self.silver_dir / relative_path.parent / name

    def _is_candidate(self, path: Path) -> bool:
        """
        Verifica se um arquivo é uma imagem a ser processada

        Args:
            path: Caminho do arquivo

        Returns:
            bool: True se for candidato à extração
        """
        if not path.is_file() or path.suffix.lower() not in self.IMAGE_EXTENSIONS:
            return False
        if path.stem.lower().endswith(".tmp"):
            return False
        is_ocr_image = path.stem.lower().endswith("_ocr")
        return is_ocr_image == self.use_ocr_images

    @staticmethod
    def _normalize_search_text(text: str) -> str:
        """
        Wrapper estático para normalizar texto de busca

        Args:
            text: Texto original

        Returns:
            str: Texto normalizado
        """
        return normalize_search_text(text)

    def _load_segment_products(self) -> list[str]:
        """
        Carrega e normaliza os produtos do segments.yaml

        Returns:
            list[str]: Produtos do segmento (normalizados)

        Raises:
            FileNotFoundError: Se o YAML não existir
            ValueError: Se o YAML for inválido ou não tiver produtos
        """
        if not self.segments_path.is_file():
            raise FileNotFoundError(
                f"Arquivo de segmentos não encontrado: {self.segments_path}"
            )

        try:
            configuration = yaml.safe_load(
                self.segments_path.read_text(encoding="utf-8")
            )
        except yaml.YAMLError as exc:
            raise ValueError(
                f"YAML inválido em {self.segments_path}: {exc}"
            ) from exc

        items = (
            configuration.get("produtos")
            if isinstance(configuration, dict) else None
        )
        if not isinstance(items, list):
            raise ValueError(
                f"A chave 'produtos' deve ser uma lista em {self.segments_path}"
            )

        products = {
            self._normalize_search_text(item["produto"])
            for item in items
            if isinstance(item, dict)
            and isinstance(item.get("produto"), str)
            and self._normalize_search_text(item["produto"])
        }
        if not products:
            raise ValueError(
                f"Nenhum produto válido encontrado em {self.segments_path}"
            )
        return sorted(products)

    # ------------------------------------------------------------------
    # OCR: acha os produtos do segmento e as suas regiões
    # ------------------------------------------------------------------

    def _load_ocr(self) -> None:
        """
        Inicializa o leitor PaddleOCR (uma única vez)

        Returns:
            None
        """
        if self._ocr_reader is not None:
            return

        from paddleocr import PaddleOCR

        self._ocr_reader = PaddleOCR(
            lang="pt",
            device="cpu",
            engine="onnxruntime",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )

    @staticmethod
    def _get_result_data(result) -> dict:
        """
        Extrai o dict com 'rec_texts'/'rec_boxes' de um resultado do OCR

        Tolera as variações entre versões/engines:
            resultado.json["res"] | resultado.json | resultado (dict)

        Args:
            result: Resultado bruto do PaddleOCR

        Returns:
            dict: Dicionário com os textos e caixas do OCR
        """
        data = None
        try:
            data = result.json
            if callable(data):
                data = data()
        except Exception:
            data = None

        if data is None and isinstance(result, dict):
            data = result

        if not isinstance(data, dict):
            return {}

        interno = data.get("res")
        if isinstance(interno, dict):
            return interno
        return data

    @staticmethod
    def _extract_boxes(result_data: dict, count: int) -> list[tuple] | None:
        """
        Extrai as caixas (x0, y0, x1, y1) de cada texto reconhecido

        Args:
            result_data: Dicionário com 'rec_texts' e 'rec_boxes'/'rec_polys'
            count: Número de textos reconhecidos

        Returns:
            list[tuple] | None: Lista de caixas ou None se não disponíveis
        """
        boxes = result_data.get("rec_boxes")
        if boxes is not None and len(boxes) == count:
            return [tuple(float(v) for v in box[:4]) for box in boxes]

        for key in ("rec_polys", "dt_polys"):
            polygons = result_data.get(key)
            if polygons is not None and len(polygons) == count:
                result = []
                for polygon in polygons:
                    xs = [float(point[0]) for point in polygon]
                    ys = [float(point[1]) for point in polygon]
                    result.append((min(xs), min(ys), max(xs), max(ys)))
                return result

        return None

    def _read_ocr(
        self, rgb_image: np.ndarray
    ) -> tuple[list[str], list[LinhaOCR], bool]:
        """
        Roda o OCR em uma imagem e retorna textos e caixas

        Args:
            rgb_image: Imagem como array numpy (RGB)

        Returns:
            tuple: (textos, linhas com caixa, posições_ok)
        """
        texts: list[str] = []
        lines: list[LinhaOCR] = []
        positions_available = True

        for result in self._ocr_reader.predict(rgb_image):
            result_data = self._get_result_data(result)
            if not result_data:
                logger.warning(
                    "Resultado do OCR em formato inesperado (tipo %s).",
                    type(result).__name__,
                )
                continue

            recognized_texts = [str(t) for t in list(result_data.get("rec_texts", []))]
            boxes = self._extract_boxes(result_data, len(recognized_texts))
            texts.extend(recognized_texts)

            if boxes is None:
                positions_available = False
                continue

            for text, (x0, y0, x1, y_end) in zip(recognized_texts, boxes):
                lines.append(LinhaOCR(
                    text=text,
                    normalized_text=self._normalize_search_text(text),
                    x0=x0, y0=y0, x1=x1, y_end=y_end,
                ))

        return texts, lines, positions_available

    def _read_ocr_strips(self, rgb_image: np.ndarray) -> list[str]:
        """
        Relê a imagem inteira em faixas horizontais sobrepostas ampliadas

        Args:
            rgb_image: Imagem como array numpy (RGB)

        Returns:
            list[str]: Textos lidos nas faixas
        """
        height = rgb_image.shape[0]
        step = max(1, int(OCR_STRIP_HEIGHT * (1 - OCR_STRIP_OVERLAP)))
        texts: list[str] = []
        y_position = 0

        while True:
            y_end = min(height, y_position + OCR_STRIP_HEIGHT)
            strip = rgb_image[y_position:y_end]

            if OCR_STRIP_SCALE != 1.0:
                image = Image.fromarray(strip)
                image = image.resize(
                    (round(image.width * OCR_STRIP_SCALE),
                     round(image.height * OCR_STRIP_SCALE)),
                    Image.Resampling.LANCZOS,
                )
                strip = np.array(image)

            texts.extend(self._read_ocr(strip)[0])

            if y_end >= height:
                break
            y_position += step

        return texts

    def _find_products(self, texts: list[str]) -> list[str]:
        """
        Acha quais produtos do segmento aparecem no texto do OCR

        Args:
            texts: Lista de textos do OCR

        Returns:
            list[str]: Produtos do segmento encontrados
        """
        ocr_text = self._normalize_search_text(" ".join(texts))
        return [
            product for product in self.segment_products
            if product_matches(product, ocr_text)
        ]

    def _find_segment_products(self, image_path: Path) -> AnaliseOCR | None:
        """
        Lê a imagem com OCR e decide se ela deve seguir para o LLM

        Args:
            image_path: Caminho da imagem

        Returns:
            AnaliseOCR | None: Análise se houver produto do segmento, None caso
                contrário (a imagem deve ser ignorada)
        """
        self._load_ocr()

        with Image.open(image_path) as image:
            rgb_image = np.array(image.convert("RGB"))
        height, width = rgb_image.shape[:2]

        self._motivo_ignorada = "sem_produto_do_segmento"

        texts, lines, positions_available = self._read_ocr(rgb_image)
        logger.info("OCR leu %d linhas de texto.", len(texts))
        if self.debug:
            logger.debug("Amostra do OCR: %s", texts[:20])

        found_products = self._find_products(texts)
        used_strips = False

        # Passada 2: nada achado -> relê a imagem toda em faixas ampliadas
        if not found_products and OCR_USE_STRIPS:
            logger.info(
                "Nenhum produto na leitura inteira; "
                "relendo em faixas ampliadas..."
            )
            strip_texts = self._read_ocr_strips(rgb_image)
            logger.info(
                "OCR em faixas leu %d linhas de texto.", len(strip_texts)
            )
            texts = texts + strip_texts
            found_products = self._find_products(texts)
            used_strips = bool(found_products)

        if not found_products:
            logger.info(
                "Nenhum produto do segmento encontrado pelo OCR. "
                "Amostra do texto: %s",
                self._normalize_search_text(" ".join(texts))[:200]
                or "(vazio)",
            )
            return None

        logger.info(
            "Produtos do segmento encontrados pelo OCR%s: %s",
            " (leitura em faixas)" if used_strips else "",
            ", ".join(found_products),
        )

        # Filtro opcional por quantidade de preços (desligado por padrão)
        price_count = sum(1 for t in texts if _looks_like_price(t))
        logger.info("Preços detectados pelo OCR: %d", price_count)
        if OCR_MIN_PRICES and price_count < OCR_MIN_PRICES:
            logger.info(
                "Poucos preços no OCR (%d < %d): não parece encarte.",
                price_count, OCR_MIN_PRICES,
            )
            self._motivo_ignorada = "poucos_precos_no_ocr"
            return None

        if used_strips:
            return AnaliseOCR(
                found_products, [], (width, height), True,
                "produto achado na leitura em faixas",
            )

        image_size = (width, height)

        if not CROP_PRODUCTS:
            return AnaliseOCR(found_products, [], image_size, True, "recorte desligado")

        if not positions_available or not lines:
            logger.warning(
                "O OCR não devolveu as posições do texto. "
                "Usando o encarte inteiro."
            )
            return AnaliseOCR(found_products, [], image_size, True, "sem posições do OCR")

        localizador = LocalizadorOfertas(lines, width, height)
        logger.debug(
            "Geometria: altura típica de linha=%.0fpx | preços detectados=%d",
            localizador.lh, len(localizador.precos),
        )

        occurrences = localizador.find_occurrences(found_products)
        regions = [
            localizador.region_for(product, idxs)
            for product, idxs in occurrences
        ]

        products_without_region = [
            product for product in found_products
            if product not in {p for r in regions for p in r.products}
        ]
        if products_without_region:
            logger.warning(
                "Não foi possível localizar na imagem: %s. "
                "Usando o encarte inteiro.", ", ".join(products_without_region),
            )
            return AnaliseOCR(
                found_products, [], image_size, True, "produto não localizado"
            )

        regions = merge_regions(regions)

        for number, region in enumerate(regions, start=1):
            logger.info(
                "Região %d: %s | caixa=%s | método=%s",
                number, ", ".join(region.products), region.box, region.method,
            )

        return AnaliseOCR(found_products, regions, image_size, False)

    # ------------------------------------------------------------------
    # Lote
    # ------------------------------------------------------------------

    def process_bronze_folder(self) -> None:
        """
        Processa todas as imagens da Bronze e encerra o servidor ao final

        Returns:
            None
        """
        try:
            self._process_bronze_folder()
        finally:
            self.close()

    def _process_bronze_folder(self) -> None:
        """
        Percorre as imagens da Bronze e processa cada uma

        Returns:
            None
        """
        batch_start_time = time.monotonic()
        self._stats = self._new_stats()

        images = sorted(
            path for path in self.bronze_dir.rglob("*")
            if self._is_candidate(path)
        )

        if not images:
            item_type = (
                "terminadas em '_ocr'"
                if self.use_ocr_images else "originais"
            )
            logger.warning(
                "Nenhuma imagem %s em %s", item_type, self.bronze_dir
            )
            return

        logger.info("Imagens encontradas na Bronze: %d", len(images))

        if self.skip_existing:
            pending_items = [
                p for p in images if not self._json_path(p).exists()
            ]
            logger.info(
                "Já processadas: %d | Pendentes: %d",
                len(images) - len(pending_items), len(pending_items),
            )
            images = pending_items

        if self.max_images is not None:
            images = images[: self.max_images]
            logger.info(
                "Limitando a %d imagem(ns) (max_images).", len(images)
            )

        if not images:
            logger.info("Nada novo para processar.")
            return

        successes = skipped_by_ocr = failures = 0
        durations: list[float] = []

        for index, image_path in enumerate(images, start=1):
            profile_name = image_path.parent.name
            start_time = time.monotonic()

            logger.info("=" * 60)
            logger.info(
                "[%d/%d] %s | perfil: %s",
                index, len(images), image_path.name, profile_name,
            )

            try:
                analysis = self._find_segment_products(image_path)
                if analysis is None:
                    logger.info("Imagem ignorada antes da chamada ao LLM.")
                    self._save_rejected_image(
                        image_path, profile_name, self._motivo_ignorada
                    )
                    skipped_by_ocr += 1
                else:
                    self._extract_data(image_path, profile_name, analysis)
                    successes += 1
            except Exception:
                failures += 1
                logger.exception(
                    "FALHA ao processar a imagem %s", image_path
                )

            durations.append(time.monotonic() - start_time)
            average_time = sum(durations) / len(durations)
            remaining_items = len(images) - index
            logger.info(
                "Tempo desta imagem: %.1fs | média: %.1fs | "
                "restantes: %d | ETA ≈ %.1f min",
                durations[-1], average_time, remaining_items, average_time * remaining_items / 60,
            )

        total_count = time.monotonic() - batch_start_time
        logger.info("=" * 60)
        logger.info(
            "RESUMO: %d sucessos | %d ignoradas pelo OCR | %d falhas | %d ofertas | "
            "%d produtos (%d compartilhadas) | %d avisos | %.1f min",
            successes, skipped_by_ocr, failures,
            self._stats["ofertas"], self._stats["produtos"],
            self._stats["compartilhadas"], self._stats["avisos"],
            total_count / 60,
        )

    # ------------------------------------------------------------------
    # Imagem: recortes e preparação
    # ------------------------------------------------------------------

    @staticmethod
    def _stack_crops(crops: list[Image.Image]) -> Image.Image:
        """
        Empilha recortes na vertical, separados por uma linha cinza

        Args:
            crops: Lista de recortes

        Returns:
            Image.Image: Imagem única com os recortes empilhados
        """
        if len(crops) == 1:
            return crops[0]

        width = max(r.width for r in crops)
        height = (
            sum(r.height for r in crops)
            + COMPOSITE_GAP * (len(crops) - 1)
        )
        tela = Image.new("RGB", (width, height), (255, 255, 255))
        desenho = ImageDraw.Draw(tela)

        y_position = 0
        for k, crop in enumerate(crops):
            tela.paste(crop, (0, y_position))
            y_position += crop.height
            if k < len(crops) - 1:
                meio = y_position + COMPOSITE_GAP // 2
                desenho.line([(0, meio), (width, meio)], fill=(150, 150, 150), width=2)
                y_position += COMPOSITE_GAP

        return tela

    def _build_crops(
        self, image: Image.Image, regions: list[RegiaoOferta]
    ) -> list[Image.Image]:
        """
        Recorta cada região e agrupa em imagens, respeitando o limite de altura

        Args:
            image: Imagem original
            regions: Regiões a recortar

        Returns:
            list[Image.Image]: Imagens prontas para enviar ao modelo
        """
        crops = [image.crop(region.box) for region in regions]

        batches: list[list[Image.Image]] = []
        current: list[Image.Image] = []
        height = 0

        for crop in crops:
            new = height + crop.height + (COMPOSITE_GAP if current else 0)
            if current and new > COMPOSITE_MAX_HEIGHT:
                batches.append(current)
                current, new = [], crop.height
            current.append(crop)
            height = new

        if current:
            batches.append(current)

        return [self._stack_crops(lote) for lote in batches]

    @staticmethod
    def _prepare_image(image: Image.Image) -> tuple[bytes, dict]:
        """
        Ajusta o tamanho da imagem e gera o PNG para envio ao modelo

        Args:
            image: Imagem original

        Returns:
            tuple[bytes, dict]: Bytes do PNG e metadados do envio
        """
        image = image.convert("RGB")
        original_name = image.size
        maior = max(image.size)

        if maior > MAX_LADO:
            image.thumbnail((MAX_LADO, MAX_LADO), Image.Resampling.LANCZOS)
        elif maior < MIN_LADO_ENVIO:
            fator = MIN_LADO_ENVIO / maior
            image = image.resize(
                (round(image.width * fator), round(image.height * fator)),
                Image.Resampling.LANCZOS,
            )

        resized_size = image.size
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        data = buffer.getvalue()
        info = {
            "original": original_name,
            "enviada": resized_size,
            "kb": len(data) / 1024,
            "tokens_estimados": (resized_size[0] // 28) * (resized_size[1] // 28),
        }
        return data, info

    # ------------------------------------------------------------------
    # Extração
    # ------------------------------------------------------------------

    def _extract_data(
        self, image_path: Path, profile_name: str, analysis: AnaliseOCR
    ) -> None:
        """
        Envia a imagem (inteira ou recortes) ao LLM e grava o JSON final

        Além da extração principal, faz uma segunda chamada ao LLM para
        casar os produtos do segments.yaml com o que aparece na imagem,
        preenchendo `produtos_encontrados` no JSON

        Args:
            image_path: Caminho da imagem
            profile_name: Nome do perfil (pasta pai da imagem)
            analysis: Resultado da análise OCR da imagem

        Returns:
            None
        """
        with Image.open(image_path) as arquivo:
            original_name = arquivo.convert("RGB")

        if analysis.use_full_image or not analysis.regions:
            images = [original_name]
            logger.info(
                "Enviando o encarte inteiro ao modelo (%s).",
                analysis.reason or "sem regiões",
            )
        else:
            images = self._build_crops(original_name, analysis.regions)
            total_area = original_name.width * original_name.height
            crop_area = sum(
                (r.box[2] - r.box[0]) * (r.box[3] - r.box[1])
                for r in analysis.regions
            )
            logger.info(
                "Recortes: %d região(ões) em %d imagem(ns) para o modelo "
                "(%.0f%% da área do encarte).",
                len(analysis.regions), len(images),
                100 * crop_area / total_area,
            )

        results: list[ResultadoExtracao] = []
        for number, image in enumerate(images, start=1):
            try:
                results.append(
                    self._query_model(image, image_path, number, len(images))
                )
            except RespostaCortada:
                logger.warning(
                    "Resposta cortada (provável loop de repetição). "
                    "Tentando de novo com amostragem diferente..."
                )
                results.append(
                    self._query_model(
                        image, image_path, number, len(images), attempt=2
                    )
                )

        # encarte_valido=true sem nenhuma oferta é contradição:
        # trata como não encarte
        valid_flyers = [r for r in results if r.encarte_valido and r.ofertas]
        if not valid_flyers:
            self._save_rejected_image(
                image_path, profile_name, "modelo_nao_confirmou_encarte",
                analysis.found_products,
            )
            logger.info("Não é encarte. Pulando para a próxima.")
            return

        result = ResultadoExtracao(
            encarte_valido=True,
            validade=next((r.validade for r in valid_flyers if r.validade), None),
            ofertas=[offer for r in valid_flyers for offer in r.ofertas],
        )

        logger.info(
            "Encarte confirmado: %d ofertas | validade: %s",
            len(result.ofertas),
            result.validade or "não informada",
        )

        lines = self._expand_offers(result)
        if not lines:
            # o modelo disse "encarte", mas nenhuma oferta com produto e
            # preço sobreviveu à validação
            self._save_rejected_image(
                image_path, profile_name, "nenhuma_oferta_valida",
                analysis.found_products,
            )
            logger.info("Nenhuma oferta válida. Tratado como não encarte.")
            return

        # Revisita com o LLM: quais produtos do YAML aparecem nesta imagem
        found_products = self._revisit_segment_products(original_name, image_path)

        self._save_json(
            image_path, profile_name, result, lines, found_products,
        )

    def _query_model(
        self,
        image: Image.Image,
        image_path: Path,
        number: int,
        total_count: int,
        attempt: int = 1,
    ) -> ResultadoExtracao:
        """
        Faz uma chamada ao modelo para uma imagem (encarte inteiro ou recortes)

        Args:
            image: Imagem a enviar
            image_path: Caminho da imagem original (para debug/logs)
            number: Número da imagem no lote atual (para logs)
            total_count: Total de imagens no lote atual (para logs)
            attempt: Número da tentativa (2 em diante ajusta a amostragem)

        Returns:
            ResultadoExtracao: Resultado validado com Pydantic

        Raises:
            RespostaCortada: Se a resposta for cortada por limite de tokens
            ValueError: Se a resposta for vazia ou inválida no schema
        """
        imagem_bytes, info = self._prepare_image(image)

        logger.info(
            "Imagem %d/%d: %dx%d -> %dx%d (%.0f KB, ≈%d tokens de visão)",
            number, total_count, *info["original"], *info["enviada"],
            info["kb"], info["tokens_estimados"],
        )

        if self.debug:
            self._save_debug_image(image_path, image, number)

        start_time = time.monotonic()
        chunks: list[str] = []
        last_chunk = None

        self._modelo_utilizado = True

        options = {
            "temperature": 0,
            "num_ctx": NUM_CTX,
            "num_predict": NUM_PREDICT,
            "num_thread": NUM_THREAD,
            "num_batch": NUM_BATCH,
        }
        if attempt > 1:
            # 2ª tentativa após resposta cortada: quebra o loop de repetição
            options.update(
                {"temperature": 0.3, "repeat_penalty": 1.1, "seed": attempt}
            )

        stream = ollama.chat(
            model=self.model_name,
            messages=[{
                "role": "user",
                "content": PROMPT,
                "images": [imagem_bytes],
            }],
            format=ResultadoExtracao.model_json_schema(),
            options=options,
            stream=True,
            keep_alive="5m",
        )

        try:
            for chunk in stream:
                if not chunks:
                    logger.info(
                        "Primeiro token após %.1fs.",
                        time.monotonic() - start_time,
                    )
                chunks.append(chunk["message"]["content"])
                last_chunk = chunk
        finally:
            close_stream = getattr(stream, "close", None)
            if callable(close_stream):
                try:
                    close_stream()
                except Exception:
                    pass

        response = "".join(chunks)
        total_duration = time.monotonic() - start_time

        # -------- Estatísticas do Ollama por imagem --------
        if last_chunk is not None:
            def _g(k):
                """Lê um campo da última resposta do Ollama."""
                return (
                    last_chunk.get(k) if isinstance(last_chunk, dict)
                    else getattr(last_chunk, k, None)
                )

            prompt_count = _g("prompt_eval_count")
            prompt_duration_ns = _g("prompt_eval_duration")
            eval_count = _g("eval_count")
            eval_duration_ns = _g("eval_duration")

            if prompt_count and prompt_duration_ns:
                logger.info(
                    "Prefill: %d tokens em %.1fs (%.1f tok/s)",
                    prompt_count, prompt_duration_ns / 1e9,
                    prompt_count / (prompt_duration_ns / 1e9),
                )
            if eval_count and eval_duration_ns:
                logger.info(
                    "Geração: %d tokens em %.1fs (%.1f tok/s)",
                    eval_count, eval_duration_ns / 1e9,
                    eval_count / (eval_duration_ns / 1e9),
                )
            logger.info("Tempo total desta chamada: %.1fs", total_duration)

        logger.debug(
            "Início da resposta bruta: %s",
            response[:300].replace("\n", " "),
        )

        if self.debug:
            self._save_debug_response(image_path, response, number)

        if last_chunk is not None:
            reason = (
                last_chunk.get("done_reason")
                if isinstance(last_chunk, dict)
                else getattr(last_chunk, "done_reason", None)
            )
            if reason == "length":
                raise RespostaCortada(
                    "Resposta cortada no limite de tokens. "
                    "Reduza a imagem, divida o encarte ou aumente "
                    "NUM_PREDICT/NUM_CTX."
                )

        if not response.strip():
            raise ValueError("O modelo retornou uma resposta vazia.")

        try:
            return ResultadoExtracao.model_validate_json(response)
        except ValidationError as exc:
            raise ValueError(
                f"Resposta do modelo não corresponde ao esquema: {exc}"
            ) from None

    def _revisit_segment_products(
        self, image: Image.Image, image_path: Path
    ) -> list[str]:
        """
        Segunda passada no LLM: casa os produtos do segments.yaml com a imagem

        Não usa os recortes: envia a imagem inteira, pois a revisita precisa
        ver o encarte todo para casar com o YAML. Em qualquer falha retorna
        lista vazia para não derrubar o processamento da imagem

        Args:
            image: Imagem inteira do encarte
            image_path: Caminho da imagem original (para logs)

        Returns:
            list[str]: Subconjunto de produtos do YAML encontrados na imagem
        """
        if not self.segment_products:
            return []

        reference_list = "\n".join(f"- {p}" for p in self.segment_products)
        prompt = PROMPT_PRODUCTS.format(lista=reference_list)

        imagem_bytes, _ = self._prepare_image(image)

        self._modelo_utilizado = True
        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[{
                    "role": "user",
                    "content": prompt,
                    "images": [imagem_bytes],
                }],
                format=ProdutosEncontrados.model_json_schema(),
                options={
                    "temperature": 0,
                    "num_ctx": NUM_CTX,
                    "num_predict": 512,
                    "num_thread": NUM_THREAD,
                    "num_batch": NUM_BATCH,
                },
                keep_alive="5m",
            )
        except Exception:
            logger.exception("Falha na revisita de produtos pelo LLM.")
            return []

        try:
            content = response["message"]["content"]
            data = ProdutosEncontrados.model_validate_json(content)
        except Exception:
            logger.exception(
                "Resposta inválida na revisita de produtos pelo LLM."
            )
            return []

        # Valida contra o YAML (case/acento-insensível) e devolve a grafia do YAML
        items_by_normalized_name = {normalize_search_text(p): p for p in self.segment_products}
        found_products: list[str] = []
        for item in data.produtos:
            key = normalize_search_text(item or "")
            if not key:
                continue
            original_name = items_by_normalized_name.get(key)
            if original_name and original_name not in found_products:
                found_products.append(original_name)

        logger.info(
            "Revisita LLM: %d produto(s) do YAML confirmados na imagem: %s",
            len(found_products), found_products or "(nenhum)",
        )
        return found_products

    # ------------------------------------------------------------------
    # Expansão
    # ------------------------------------------------------------------

    def _expand_offers(
        self, result: ResultadoExtracao
    ) -> list[dict]:
        """
        Expande cada oferta em uma linha por variante de produto

        Valida preços e coerência de condição, e remove duplicatas

        Args:
            result: Resultado da extração do modelo

        Returns:
            list[dict]: Linhas de produto prontas para o JSON final
        """
        lines: list[dict] = []
        seen_items: set[tuple] = set()

        for number, offer in enumerate(result.ofertas, start=1):
            variants = [v for v in offer.variantes if v.produto]

            if not variants:
                logger.warning(
                    "Oferta %d sem produto. Descartada.", number
                )
                self._stats["avisos"] += 1
                continue

            if (
                offer.preco_atual is None
                and offer.preco_anterior is None
                and offer.preco_condicao is None
            ):
                logger.warning(
                    "Oferta %d (%s) sem preço. Descartada.",
                    number, variants[0].produto,
                )
                self._stats["avisos"] += 1
                continue

            if (
                offer.preco_atual is not None
                and offer.preco_anterior is not None
                and offer.preco_anterior <= offer.preco_atual
            ):
                logger.warning(
                    "Oferta %d (%s): preco_anterior (%.2f) <= atual (%.2f).",
                    number, variants[0].produto,
                    offer.preco_anterior, offer.preco_atual,
                )
                self._stats["avisos"] += 1

            if (
                offer.preco_atual is not None
                and offer.preco_condicao is not None
                and offer.preco_condicao > offer.preco_atual
            ):
                logger.warning(
                    "Oferta %d (%s): preco_condicao (%.2f) > atual (%.2f).",
                    number, variants[0].produto,
                    offer.preco_condicao, offer.preco_atual,
                )
                self._stats["avisos"] += 1

            # Coerência condição <-> preço condição
            if (offer.condicao is None) != (offer.preco_condicao is None):
                logger.warning(
                    "Oferta %d (%s): condicao=%r e preco_condicao=%r "
                    "incoerentes (um sem o outro).",
                    number, variants[0].produto,
                    offer.condicao, offer.preco_condicao,
                )
                self._stats["avisos"] += 1

            is_shared_offer = len(variants) > 1
            self._stats["ofertas"] += 1
            if is_shared_offer:
                self._stats["compartilhadas"] += 1
                logger.info(
                    "Oferta %d: %d produtos pelo MESMO preço (R$ %s): %s",
                    number, len(variants), offer.preco_atual,
                    " | ".join(v.produto for v in variants),
                )
            else:
                condition_details = (
                    f" | R$ {offer.preco_condicao} "
                    f"({offer.condicao or 'condição'})"
                    if offer.preco_condicao is not None else ""
                )
                logger.info(
                    "Oferta %d: %s | R$ %s%s",
                    number, variants[0].produto,
                    offer.preco_atual, condition_details,
                )

            for variant in variants:
                key = (
                    variant.produto.casefold(),
                    variant.quantidade,
                    variant.unidade_medida,
                    offer.preco_atual,
                )
                if key in seen_items:
                    logger.debug("Duplicado ignorado: %s", variant.produto)
                    continue
                seen_items.add(key)

                lines.append({
                    "oferta_compartilhada": is_shared_offer,
                    "variantes_na_oferta": len(variants),
                    "produto": variant.produto,
                    "marca": variant.marca,
                    "quantidade": variant.quantidade,
                    "unidade_medida": variant.unidade_medida,
                    "preco_atual": offer.preco_atual,
                    "preco_anterior": offer.preco_anterior,
                    "preco_condicao": offer.preco_condicao,
                    "condicao": offer.condicao,
                })

        self._stats["produtos"] += len(lines)
        logger.info(
            "Resultado: %d ofertas -> %d linhas de produto.",
            len(result.ofertas), len(lines),
        )
        return lines

    # ------------------------------------------------------------------
    # Saída em disco
    # ------------------------------------------------------------------

    def _save_debug_response(self, image_path: Path, response: str, number: int) -> None:
        """
        Salva a resposta bruta do modelo em um arquivo de debug

        Args:
            image_path: Caminho da imagem
            response: Resposta bruta do modelo
            number: Número da imagem no lote

        Returns:
            None
        """
        base = self._json_path(image_path)
        destino = base.with_name(f"{base.stem}.debug_{number}.txt")
        destino.parent.mkdir(parents=True, exist_ok=True)
        destino.write_text(response, encoding="utf-8")
        logger.debug("Resposta bruta salva em: %s", destino)

    def _save_debug_image(
        self, image_path: Path, image: Image.Image, number: int
    ) -> None:
        """
        Salva exatamente a imagem enviada ao modelo (para debug)

        Args:
            image_path: Caminho da imagem
            image: Imagem enviada ao modelo
            number: Número da imagem no lote

        Returns:
            None
        """
        base = self._json_path(image_path)
        destino = base.with_name(f"{base.stem}.recorte_{number}.png")
        destino.parent.mkdir(parents=True, exist_ok=True)
        image.save(destino, format="PNG")
        logger.debug("Imagem enviada ao modelo salva em: %s", destino)

    def _save_json(
        self, image_path: Path, profile_name: str,
        result: ResultadoExtracao, lines: list[dict],
        found_products: list[str],
    ) -> Path:
        """
        Grava o JSON final da extração (escrita atômica via .tmp)

        Args:
            image_path: Caminho da imagem
            profile_name: Nome do perfil
            result: Resultado validado da extração
            lines: Linhas de produto já expandidas
            found_products: Produtos do YAML confirmados pelo LLM

        Returns:
            Path: Caminho do JSON gravado
        """
        relative_path = image_path.relative_to(self.bronze_dir)
        output_path = self._json_path(image_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        resultado_final = {
            "encarte_valido": True,
            "perfil": profile_name,
            "arquivo_origem": str(relative_path),
            "data_processamento": date.today().isoformat(),
            "modelo": self.model_name,
            "validade": result.validade,
            "produtos_encontrados": found_products,
            "quantidade_produtos": len(lines),
            "produtos": lines,
        }

        temp_path = output_path.with_suffix(".tmp")
        try:
            temp_path.write_text(
                json.dumps(resultado_final, ensure_ascii=False, indent=4),
                encoding="utf-8",
            )
            temp_path.replace(output_path)
        finally:
            if temp_path.exists():
                temp_path.unlink()

        logger.info("Salvo: %s", output_path)
        return output_path

    def _save_rejected_image(
        self,
        image_path: Path,
        profile_name: str,
        reason: str,
        found_products: list[str] | None = None,
    ) -> Path:
        """
        Grava o JSON de uma imagem que não é (ou não interessa como) encarte

        O JSON fica com encarte_valido=false, para que a imagem não seja
        reprocessada nas próximas execuções (skip_existing) e para registrar
        o motivo

        Args:
            image_path: Caminho da imagem
            profile_name: Nome do perfil
            reason: Motivo pelo qual a imagem foi descartada
            found_products: Produtos do segmento que o OCR encontrou

        Returns:
            Path: Caminho do JSON gravado
        """
        output_path = self._json_path(image_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "encarte_valido": False,
            "motivo": reason,
            "perfil": profile_name,
            "arquivo_origem": str(image_path.relative_to(self.bronze_dir)),
            "data_processamento": date.today().isoformat(),
            "validade": None,
            "produtos_encontrados": found_products or [],
            "quantidade_produtos": 0,
            "produtos": [],
        }

        temp_path = output_path.with_suffix(".tmp")
        try:
            temp_path.write_text(
                json.dumps(data, ensure_ascii=False, indent=4),
                encoding="utf-8",
            )
            temp_path.replace(output_path)
        finally:
            if temp_path.exists():
                temp_path.unlink()
        logger.info("Não encarte (%s) | %s", reason, output_path)
        return output_path

    # ------------------------------------------------------------------
    # Encerramento
    # ------------------------------------------------------------------

    def close(self) -> None:
        """
        Libera os recursos da automação (Playwright, Ollama, processo)

        Returns:
            None
        """
        if self._fechando or self._fechado:
            return
        self._fechando = True
        try:
            usou_modelo = self._modelo_utilizado
            self._modelo_utilizado = False

            should_terminate_server = (
                self._ollama_process is not None
                or (usou_modelo and TERMINATE_EXISTING_SERVER)
            )

            if usou_modelo and not should_terminate_server:
                logger.info(
                    "Solicitando descarregamento do modelo %s...",
                    self.model_name,
                )
                try:
                    ollama.generate(
                        model=self.model_name, prompt="", keep_alive=0,
                    )
                    logger.info("Solicitação de descarregamento enviada.")
                except Exception:
                    logger.warning(
                        "Não foi possível solicitar o descarregamento."
                    )

            if self._ollama_process is not None:
                self._stop_owned_server()
            elif usou_modelo and TERMINATE_EXISTING_SERVER:
                self._stop_existing_server()

            self._fechado = True
        finally:
            self._fechando = False

    def __enter__(self):
        """
        Inicializa o extrator ao entrar em um bloco `with`

        Returns:
            LLMExtractor: Instância pronta para uso
        """
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Fecha os recursos ao sair de um bloco `with`

        Args:
            exc_type: Tipo da exceção ocorrida no bloco, se houver
            exc_value: Instância da exceção ocorrida, se houver
            traceback: Rastreamento da exceção, se houver

        Returns:
            bool: False (propaga a exceção, se houver)
        """
        self.close()
        return False