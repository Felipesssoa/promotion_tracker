"""Consolidação da camada Silver reviewed na camada Gold.

Lê os JSONs normalizados, descarta encartes inválidos e
mantém apenas as linhas de produto cujo campo ``produto`` casa com algum
item de ``config/product.yaml``. O resultado é gravado como um único JSON
em ``data/gold/AAAA/MM/DD/ofertas_consolidadas.json`` (escrita atômica) e
uma cópia em ``data/gold/latest.json``.
"""

from __future__ import annotations

import json
import logging
import re
import unicodedata
from datetime import datetime
from pathlib import Path

import yaml

from ..common.json_io import write_json_atomic
from ..common.paths import PROJECT_PATHS, ProjectPaths

logger = logging.getLogger(__name__)

_PUNCT = re.compile(r"[^\w\s]")


def _normalize(text: str) -> str:
    """Remove acentos, caixa e pontuação para comparação tolerante."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = _PUNCT.sub(" ", text)
    return " ".join(text.casefold().split())


class Consolidator:
    """Agrega Silver reviewed em um JSON restrito aos produtos monitorados.

    Todos os caminhos têm default resolvido a partir da raiz do projeto,
    então ``Consolidator()`` funciona sem argumentos.

    Args:
        silver_dir: Raiz dos JSONs revisados. Default: ``<root>/data/silver_reviewed``.
        gold_dir: Raiz dos JSONs consolidados. Default: ``<root>/data/gold``.
        products_path: YAML com a lista ``produtos``/``produto``.
            Default: ``<root>/config/product.yaml``.
        paths: Caminhos do projeto; permite substituir a raiz padrão.
    """

    def __init__(
        self,
        silver_dir: Path | None = None,
        gold_dir: Path | None = None,
        products_path: Path | None = None,
        paths: ProjectPaths | None = None,
    ) -> None:
        project_paths = paths or PROJECT_PATHS
        self.silver_dir = (
            Path(silver_dir) if silver_dir else project_paths.silver_reviewed_dir
        )
        self.gold_dir = Path(gold_dir) if gold_dir else project_paths.gold_dir
        self.products_path = (
            Path(products_path)
            if products_path
            else project_paths.product_config
        )

    # ------------------------------------------------------------------ #
    # API pública
    # ------------------------------------------------------------------ #
    def consolidate(self) -> Path:
        """Executa a consolidação e retorna o caminho do JSON gerado."""
        targets = self._load_targets()

        encartes: list[dict] = []
        total_lidos = 0
        total_invalidos = 0
        total_sem_match = 0
        total_linhas = 0
        contagem: dict[str, int] = {}

        for json_path in self._iter_silver_files():
            total_lidos += 1
            record = self._read_json(json_path)
            if record is None:
                continue
            if not record.get("encarte_valido"):
                total_invalidos += 1
                continue

            relevantes = [
                linha for linha in record.get("produtos", [])
                if self._matches_target(linha.get("produto", ""), targets)
            ]
            if not relevantes:
                total_sem_match += 1
                continue

            total_linhas += len(relevantes)
            for linha in relevantes:
                nome = _normalize(str(linha.get("produto", "")))
                contagem[nome] = contagem.get(nome, 0) + 1

            try:
                arquivo_silver = str(json_path.relative_to(self.silver_dir))
            except ValueError:
                arquivo_silver = str(json_path)

            encartes.append({
                "perfil": record.get("perfil"),
                "arquivo_silver": arquivo_silver,
                "arquivo_origem": record.get("arquivo_origem"),
                "data_processamento": record.get("data_processamento"),
                "modelo": record.get("modelo"),
                "validade": record.get("validade"),
                "produtos_encontrados": record.get("produtos_encontrados", []),
                "produtos": relevantes,
            })

        now = datetime.now()
        output_dir = self.gold_dir / f"{now:%Y}" / f"{now:%m}" / f"{now:%d}"
        output_path = output_dir / "ofertas_consolidadas.json"

        payload = {
            "gerado_em": now.isoformat(timespec="seconds"),
            "product_path": str(self.products_path),
            "silver_dir": str(self.silver_dir),
            "produtos_alvo": targets,
            "resumo": {
                "jsons_lidos": total_lidos,
                "jsons_invalidos": total_invalidos,
                "jsons_sem_match": total_sem_match,
                "encartes_com_produtos": len(encartes),
                "linhas_produto": total_linhas,
                "contagem_por_produto": dict(sorted(contagem.items())),
            },
            "encartes": encartes,
        }

        self._write_json_atomic(output_path, payload)
        self._write_json_atomic(self.gold_dir / "latest.json", payload)
        logger.info(
            "Gold: %s | encartes=%d linhas=%d",
            output_path, len(encartes), total_linhas,
        )
        return output_path

    # ------------------------------------------------------------------ #
    # Internos
    # ------------------------------------------------------------------ #
    def _load_targets(self) -> list[str]:
        if not self.products_path.exists():
            logger.warning("product.yaml não encontrado: %s", self.products_path)
            return []
        with self.products_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        targets = [
            _normalize(str(item["produto"]))
            for item in data.get("produtos", [])
            if isinstance(item, dict) and "produto" in item
        ]
        targets = [t for t in targets if t]
        if not targets:
            logger.warning("Nenhum produto em %s", self.products_path)
        return targets

    def _iter_silver_files(self):
        if not self.silver_dir.exists():
            logger.warning("Silver reviewed inexistente: %s", self.silver_dir)
            return []
        return sorted(self.silver_dir.rglob("*.json"))

    @staticmethod
    def _read_json(path: Path) -> dict | None:
        try:
            with path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Falha ao ler %s: %s", path, exc)
            return None

    @staticmethod
    def _matches_target(nome: str, targets: list[str]) -> bool:
        n = _normalize(nome)
        if not n:
            return False
        return any(n == t or t in n or n in t for t in targets)

    @staticmethod
    def _write_json_atomic(output_path: Path, data: dict) -> None:
        write_json_atomic(output_path, data)