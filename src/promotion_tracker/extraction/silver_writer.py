"""Expansão e persistência dos resultados da extração.

Este módulo transforma ofertas validadas em linhas de produto, atualiza as
estatísticas do lote e grava JSONs e artefatos de debug. A escrita dos JSONs é
atômica: primeiro cria-se um arquivo temporário no mesmo diretório e, depois,
ele substitui o destino final.

Regra central da expansão: **cada produto do encarte vira no máximo UMA linha
no JSON**. O modelo pode ler o mesmo produto várias vezes (releitura em faixas,
sobreposição, alucinação de ofertas repetidas), mas a gravação deduplica por
identidade do produto (nome impresso + medida), independentemente do preço.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from datetime import date
from difflib import SequenceMatcher
from pathlib import Path

from PIL import Image

from ..common.json_io import write_json_atomic
from ..common.products import ProductCatalog
from .llm.schemas import ResultadoExtracao

logger = logging.getLogger(__name__)

# Similaridade mínima (0-1) entre dois nomes para considerá-los o MESMO
# produto lido duas vezes. Medidas diferentes (400g x 500g) nunca se fundem.
SIMILARITY_THRESHOLD = 0.92

_RE_UNIT_GLUE = re.compile(r"(\d)\s+(kg|mg|g|ml|l|lt)\b")
_RE_MEASURE = re.compile(
    r"(?<![\w,.])(\d+(?:[.,]\d+)?)\s*"
    r"(kg|mg|ml|gramas?|quilos?|litros?|lt|g|l)(?![a-zà-ÿ0-9])",
    re.IGNORECASE,
)
_UNIT_MAP = {
    "kg": "kg", "quilo": "kg", "quilos": "kg",
    "mg": "mg",
    "g": "g", "grama": "g", "gramas": "g",
    "ml": "ml",
    "l": "l", "lt": "l", "litro": "l", "litros": "l",
}


def normalize_identity(text: str) -> str:
    """Normaliza o nome de um produto para comparar identidades.

    Remove acentos, caixa e pontuação; junta número e unidade ("1,5 L" e
    "1.5l" viram ``15l``).

    Args:
        text: Nome do produto como lido no encarte.

    Returns:
        Texto normalizado, só com letras minúsculas, dígitos e espaços.
    """
    text = unicodedata.normalize("NFKD", text or "")
    text = "".join(ch for ch in text if not unicodedata.combining(ch)).casefold()
    text = re.sub(r"(\d)[.,](\d)", r"\1\2", text)
    text = _RE_UNIT_GLUE.sub(r"\1\2", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return text.strip()


def same_product(first: str, second: str) -> bool:
    """Diz se dois nomes normalizados são o mesmo produto.

    Args:
        first: Nome normalizado por ``normalize_identity``.
        second: Nome normalizado por ``normalize_identity``.

    Returns:
        ``True`` se forem iguais, tiverem as mesmas palavras ou forem muito
        parecidos. Se ambos citam medidas e elas diferem, são produtos
        diferentes.
    """
    if not first or not second:
        return False
    if first == second:
        return True
    tokens_a, tokens_b = first.split(), second.split()
    numbers_a = {t for t in tokens_a if any(c.isdigit() for c in t)}
    numbers_b = {t for t in tokens_b if any(c.isdigit() for c in t)}
    if numbers_a and numbers_b and numbers_a != numbers_b:
        return False
    if set(tokens_a) == set(tokens_b):
        return True
    return SequenceMatcher(None, first, second).ratio() >= SIMILARITY_THRESHOLD


def has_repeated_products(result: ResultadoExtracao) -> bool:
    """Indica se o mesmo produto aparece em ofertas diferentes.

    Numa leitura única do encarte isso é sinal de alucinação/loop do modelo.

    Args:
        result: Resultado validado da extração.

    Returns:
        ``True`` se algum produto aparece em mais de uma oferta.
    """
    seen: list[tuple[str, int]] = []
    for index, offer in enumerate(result.ofertas):
        for variant in offer.variantes:
            if not variant.produto:
                continue
            identity = normalize_identity(variant.produto)
            for previous, previous_index in seen:
                if previous_index != index and same_product(previous, identity):
                    return True
            seen.append((identity, index))
    return False


def measure_from_name(text: str) -> tuple[float, str] | None:
    """Extrai peso/volume escrito no nome do produto.

    Args:
        text: Nome do produto (ex.: ``"Doce de Leite Pote 320g"``).

    Returns:
        ``(quantidade, unidade)`` se houver exatamente UMA medida no nome;
        ``None`` se não houver ou se houver mais de uma (ambíguo).
    """
    matches = _RE_MEASURE.findall(text or "")
    if len(matches) != 1:
        return None
    number, unit = matches[0]
    unit = _UNIT_MAP.get(unit.lower())
    try:
        value = float(number.replace(",", "."))
    except ValueError:
        return None
    if unit is None or value <= 0:
        return None
    return value, unit


@dataclass
class _Candidate:
    """Uma variante lida do modelo, antes da deduplicação."""

    order: int
    offer_number: int
    offer: object
    variant: object
    shared_size: int
    warnings: int
    identity: str


class SilverWriter:
    """Converte resultados do modelo e grava arquivos na camada Silver.

    Args:
        bronze_dir: Raiz das imagens de entrada.
        silver_dir: Raiz dos JSONs de saída.
        model_name: Nome do modelo usado, incluído no JSON final.
        product_catalog: Catálogo usado para gravar produtos pelo nome canônico.
    """

    def __init__(
        self,
        bronze_dir: Path,
        silver_dir: Path,
        model_name: str,
        product_catalog: ProductCatalog | None = None,
    ) -> None:
        """Inicializa o escritor com as raízes de entrada e saída.

        Args:
            bronze_dir: Diretório raiz da camada Bronze.
            silver_dir: Diretório raiz da camada Silver.
            model_name: Identificador do modelo para registrar nos resultados.
            product_catalog: Alias configurado para mapear produtos extraídos.
        """
        self.bronze_dir = bronze_dir
        self.silver_dir = silver_dir
        self.model_name = model_name
        self.product_catalog = product_catalog or ProductCatalog()

    def json_path(self, image_path: Path) -> Path:
        """Calcula o JSON de saída correspondente a uma imagem Bronze.

        Mantém a hierarquia relativa à Bronze e remove o sufixo ``_ocr`` do
        nome para que a imagem original e sua versão OCR apontem ao mesmo JSON.

        Args:
            image_path: Caminho da imagem dentro da camada Bronze.

        Returns:
            Caminho do JSON correspondente dentro da camada Silver.
        """
        relative_path = image_path.relative_to(self.bronze_dir)
        name = f"{image_path.stem.removesuffix('_ocr')}.json"
        return self.silver_dir / relative_path.parent / name

    @staticmethod
    def _check_offer(
        number: int, offer, variants: list, stats: dict[str, int]
    ) -> int | None:
        """Valida uma oferta e conta os avisos de incoerência.

        Args:
            number: Posição (1-based) da oferta na resposta do modelo.
            offer: Oferta devolvida pelo modelo.
            variants: Variantes da oferta que têm nome de produto.
            stats: Contadores do lote, atualizados in-place.

        Returns:
            Quantidade de avisos da oferta, ou ``None`` se ela deve ser
            descartada (sem produto ou sem qualquer preço).
        """
        if not variants:
            logger.warning("Oferta %d sem produto. Descartada.", number)
            stats["avisos"] += 1
            return None

        if (
            offer.preco_atual is None
            and offer.preco_anterior is None
            and offer.preco_condicao is None
        ):
            logger.warning(
                "Oferta %d (%s) sem preço. Descartada.",
                number, variants[0].produto,
            )
            stats["avisos"] += 1
            return None

        warnings = 0
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
            warnings += 1

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
            warnings += 1

        if (offer.condicao is None) != (offer.preco_condicao is None):
            logger.warning(
                "Oferta %d (%s): condicao=%r e preco_condicao=%r "
                "incoerentes (um sem o outro).",
                number, variants[0].produto,
                offer.condicao, offer.preco_condicao,
            )
            warnings += 1

        stats["avisos"] += warnings
        return warnings

    @staticmethod
    def _pick_unique(
        candidates: list[_Candidate], stats: dict[str, int]
    ) -> list[_Candidate]:
        """Mantém uma única leitura de cada produto do encarte.

        Agrupa candidatos que são o mesmo produto (ignorando o preço) e fica
        com o melhor de cada grupo: o de oferta com menos variantes (leitura
        mais específica), depois o com menos avisos, depois o primeiro lido.

        Args:
            candidates: Todas as variantes lidas, na ordem de aparição.
            stats: Contadores do lote (``duplicados`` é incrementado).

        Returns:
            Candidatos vencedores, na ordem de aparição.
        """
        groups: list[list[_Candidate]] = []
        for candidate in candidates:
            for group in groups:
                if same_product(group[0].identity, candidate.identity):
                    group.append(candidate)
                    break
            else:
                groups.append([candidate])

        winners: list[_Candidate] = []
        for group in groups:
            winner = min(
                group, key=lambda c: (c.shared_size, c.warnings, c.order)
            )
            winners.append(winner)
            for other in group:
                if other is winner:
                    continue
                stats["duplicados"] = stats.get("duplicados", 0) + 1
                logger.info(
                    "Produto repetido ignorado: %s (oferta %d, R$ %s); "
                    "mantida a oferta %d (R$ %s).",
                    other.variant.produto, other.offer_number,
                    other.offer.preco_atual,
                    winner.offer_number, winner.offer.preco_atual,
                )
        winners.sort(key=lambda c: c.order)
        return winners

    @staticmethod
    def _resolve_measure(variant) -> tuple[float | None, str | None]:
        """Escolhe quantidade/unidade, priorizando a medida escrita no nome.

        Evita casos como "Pote 320g" gravado como 1 un. Só sobrescreve o
        valor do modelo quando o nome traz exatamente uma medida.

        Args:
            variant: Variante devolvida pelo modelo.

        Returns:
            Tupla ``(quantidade, unidade_medida)``.
        """
        quantity, unit = variant.quantidade, variant.unidade_medida
        measured = measure_from_name(variant.produto)
        if measured and (quantity, unit) != measured:
            logger.debug(
                "Medida corrigida pelo nome (%s): (%s, %s) -> %s",
                variant.produto, quantity, unit, measured,
            )
            return measured
        return quantity, unit

    def expand_offers(
        self, result: ResultadoExtracao, stats: dict[str, int]
    ) -> list[dict]:
        """Valida ofertas e expande cada variante em uma linha de produto.

        Ofertas sem produto ou sem qualquer preço são descartadas. Incoerências
        de preço e condição são contabilizadas como avisos. Cada produto do
        encarte é gravado UMA vez: leituras repetidas (mesmo nome e medida) são
        descartadas mesmo com preço diferente. A flag de oferta compartilhada
        reflete apenas as variantes que de fato foram gravadas.

        Args:
            result: Resposta estruturada validada pelo modelo Pydantic.
            stats: Contadores compartilhados do lote, atualizados in-place.

        Returns:
            Linhas de produto prontas para serialização no JSON final.
        """
        candidates: list[_Candidate] = []
        for number, offer in enumerate(result.ofertas, start=1):
            variants = [variant for variant in offer.variantes if variant.produto]
            warnings = self._check_offer(number, offer, variants, stats)
            if warnings is None:
                continue
            for variant in variants:
                candidates.append(_Candidate(
                    order=len(candidates),
                    offer_number=number,
                    offer=offer,
                    variant=variant,
                    shared_size=len(variants),
                    warnings=warnings,
                    identity=normalize_identity(variant.produto),
                ))

        winners = self._pick_unique(candidates, stats)
        survivors = Counter(c.offer_number for c in winners)

        lines: list[dict] = []
        names_by_offer: dict[int, list[str]] = {}
        offer_by_number: dict[int, object] = {}
        for candidate in winners:
            offer, variant = candidate.offer, candidate.variant
            is_shared_offer = survivors[candidate.offer_number] > 1
            reviewed_product = self.product_catalog.normalize_name(variant.produto)
            quantity, unit = self._resolve_measure(variant)
            names_by_offer.setdefault(candidate.offer_number, []).append(
                variant.produto
            )
            offer_by_number[candidate.offer_number] = offer
            lines.append({
                "oferta_compartilhada": is_shared_offer,
                "variantes_na_oferta": survivors[candidate.offer_number],
                "produto": reviewed_product,
                "produto_original": variant.produto,
                "produto_revisado": reviewed_product,
                "marca": variant.marca,
                "quantidade": quantity,
                "unidade_medida": unit,
                "preco_atual": offer.preco_atual,
                "preco_anterior": offer.preco_anterior,
                "preco_condicao": offer.preco_condicao,
                "condicao": offer.condicao,
            })

        for number in sorted(names_by_offer):
            offer = offer_by_number[number]
            names = names_by_offer[number]
            stats["ofertas"] += 1
            if len(names) > 1:
                stats["compartilhadas"] += 1
                logger.info(
                    "Oferta %d: %d produtos pelo MESMO preço (R$ %s): %s",
                    number, len(names), offer.preco_atual, " | ".join(names),
                )
            else:
                condition_details = (
                    f" | R$ {offer.preco_condicao} "
                    f"({offer.condicao or 'condição'})"
                    if offer.preco_condicao is not None else ""
                )
                logger.info(
                    "Oferta %d: %s | R$ %s%s",
                    number, names[0], offer.preco_atual, condition_details,
                )

        stats["produtos"] += len(lines)
        logger.info(
            "Resultado: %d ofertas lidas -> %d linhas de produto "
            "(%d repetições descartadas).",
            len(result.ofertas), len(lines), len(candidates) - len(lines),
        )
        return lines

    def save_debug_response(
        self, image_path: Path, response: str, number: int
    ) -> None:
        """Salva a resposta bruta do Ollama para inspeção.

        Args:
            image_path: Imagem de origem da chamada.
            response: Texto integral devolvido pelo modelo.
            number: Número da imagem ou recorte no lote.
        """
        base = self.json_path(image_path)
        destino = base.with_name(f"{base.stem}.debug_{number}.txt")
        destino.parent.mkdir(parents=True, exist_ok=True)
        destino.write_text(response, encoding="utf-8")
        logger.debug("Resposta bruta salva em: %s", destino)

    def save_debug_image(
        self, image_path: Path, image: Image.Image, number: int
    ) -> None:
        """Salva a imagem efetivamente enviada ao modelo.

        Args:
            image_path: Imagem de origem da chamada.
            image: Imagem preparada ou recorte enviado ao modelo.
            number: Número da imagem ou recorte no lote.
        """
        base = self.json_path(image_path)
        destino = base.with_name(f"{base.stem}.recorte_{number}.png")
        destino.parent.mkdir(parents=True, exist_ok=True)
        image.save(destino, format="PNG")
        logger.debug("Imagem enviada ao modelo salva em: %s", destino)

    def save_result(
        self,
        image_path: Path,
        profile_name: str,
        result: ResultadoExtracao,
        lines: list[dict],
        found_products: list[str],
    ) -> Path:
        """Grava o resultado válido em JSON com substituição atômica.

        Args:
            image_path: Imagem de origem na camada Bronze.
            profile_name: Identificador do perfil/rede de origem.
            result: Resultado validado da extração.
            lines: Produtos expandidos e validados.
            found_products: Produtos do segmento confirmados pelo modelo.

        Returns:
            Caminho do JSON gravado na camada Silver.
        """
        relative_path = image_path.relative_to(self.bronze_dir)
        output_path = self.json_path(image_path)
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
        self._write_json_atomic(output_path, resultado_final)
        logger.info("Salvo: %s", output_path)
        return output_path

    def save_rejected(
        self,
        image_path: Path,
        profile_name: str,
        reason: str,
        found_products: list[str] | None = None,
    ) -> Path:
        """Grava o registro de uma imagem descartada como encarte.

        O JSON negativo impede reprocessamento quando ``skip_existing`` está
        ligado e preserva a razão da decisão junto à imagem de origem.

        Args:
            image_path: Imagem de origem na camada Bronze.
            profile_name: Identificador do perfil/rede de origem.
            reason: Código que explica por que a imagem foi descartada.
            found_products: Produtos que o OCR encontrou, se houver.

        Returns:
            Caminho do JSON de rejeição gravado na camada Silver.
        """
        output_path = self.json_path(image_path)
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
        self._write_json_atomic(output_path, data)
        logger.info("Não encarte (%s) | %s", reason, output_path)
        return output_path

    @staticmethod
    def _write_json_atomic(output_path: Path, data: dict) -> None:
        """Escreve JSON UTF-8 em temporário e substitui o destino final.

        Args:
            output_path: Caminho do JSON definitivo.
            data: Conteúdo serializável em JSON.
        """
        write_json_atomic(output_path, data)