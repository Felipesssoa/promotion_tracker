"""Expansão e persistência dos resultados da extração.

Este módulo transforma ofertas validadas em linhas de produto, atualiza as
estatísticas do lote e grava JSONs e artefatos de debug. A escrita dos JSONs é
atômica: primeiro cria-se um arquivo temporário no mesmo diretório e, depois,
ele substitui o destino final.
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

from PIL import Image

from ..core.json_io import write_json_atomic
from .llm.schemas import ResultadoExtracao

logger = logging.getLogger(__name__)


class SilverWriter:
    """Converte resultados do modelo e grava arquivos na camada Silver.

    Args:
        bronze_dir: Raiz das imagens de entrada.
        silver_dir: Raiz dos JSONs de saída.
        model_name: Nome do modelo usado, incluído no JSON final.
    """

    def __init__(
        self, bronze_dir: Path, silver_dir: Path, model_name: str
    ) -> None:
        """Inicializa o escritor com as raízes de entrada e saída.

        Args:
            bronze_dir: Diretório raiz da camada Bronze.
            silver_dir: Diretório raiz da camada Silver.
            model_name: Identificador do modelo para registrar nos resultados.
        """
        self.bronze_dir = bronze_dir
        self.silver_dir = silver_dir
        self.model_name = model_name

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

    def expand_offers(
        self, result: ResultadoExtracao, stats: dict[str, int]
    ) -> list[dict]:
        """Valida ofertas e expande cada variante em uma linha de produto.

        Ofertas sem produto ou sem qualquer preço são descartadas. Incoerências
        de preço e condição são contabilizadas como avisos. Duplicatas são
        removidas usando produto, quantidade, unidade e preço atual.

        Args:
            result: Resposta estruturada validada pelo modelo Pydantic.
            stats: Contadores compartilhados do lote, atualizados in-place.

        Returns:
            Linhas de produto prontas para serialização no JSON final.
        """
        lines: list[dict] = []
        seen_items: set[tuple] = set()

        for number, offer in enumerate(result.ofertas, start=1):
            variants = [variant for variant in offer.variantes if variant.produto]
            if not variants:
                logger.warning("Oferta %d sem produto. Descartada.", number)
                stats["avisos"] += 1
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
                stats["avisos"] += 1
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
                stats["avisos"] += 1

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
                stats["avisos"] += 1

            if (offer.condicao is None) != (offer.preco_condicao is None):
                logger.warning(
                    "Oferta %d (%s): condicao=%r e preco_condicao=%r "
                    "incoerentes (um sem o outro).",
                    number, variants[0].produto,
                    offer.condicao, offer.preco_condicao,
                )
                stats["avisos"] += 1

            is_shared_offer = len(variants) > 1
            stats["ofertas"] += 1
            if is_shared_offer:
                stats["compartilhadas"] += 1
                logger.info(
                    "Oferta %d: %d produtos pelo MESMO preço (R$ %s): %s",
                    number, len(variants), offer.preco_atual,
                    " | ".join(variant.produto for variant in variants),
                )
            else:
                condition_details = (
                    f" | R$ {offer.preco_condicao} "
                    f"({offer.condicao or 'condição'})"
                    if offer.preco_condicao is not None else ""
                )
                logger.info(
                    "Oferta %d: %s | R$ %s%s",
                    number, variants[0].produto, offer.preco_atual, condition_details,
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

        stats["produtos"] += len(lines)
        logger.info(
            "Resultado: %d ofertas -> %d linhas de produto.",
            len(result.ofertas), len(lines),
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