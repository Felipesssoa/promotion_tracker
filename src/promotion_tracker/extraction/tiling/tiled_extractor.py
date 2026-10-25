"""Extrator de encartes por tiles: Bronze -> Silver.

Fluxo por imagem:
  1. Detector (gate): sem preço na imagem, não é encarte e nem chega ao modelo.
  2. Leitura por recortes com o prompt e o schema do extrator original.
  3. Marcas ausentes são completadas pela lista de config/brands.yaml.
  4. Ofertas repetidas entre recortes vizinhos são unidas.
  5. Sem nenhuma oferta com preço, a imagem vira "não encarte".
  6. O SilverWriter do projeto valida, converte aliases e grava o JSON no formato
     que o Normalizer e o Consolidator esperam.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from promotion_tracker.common.products import ProductCatalog
from promotion_tracker.extraction.llm.schemas import ResultadoExtracao
from promotion_tracker.extraction.silver_writer import SilverWriter

from .brands import BrandCatalog
from .dedup import merge_offers
from .gate import AlwaysPassGate, FlyerGate
from .tile_prompt import build_prompt, fill_missing_brands, most_common, parse_tile_response
from .tiler import split_image
from .vision_backends import VisionBackend

log = logging.getLogger(__name__)

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


def _new_stats() -> dict[str, int]:
    return {"ofertas": 0, "produtos": 0, "compartilhadas": 0, "avisos": 0}


class TiledExtractor:
    def __init__(
        self,
        backend: VisionBackend,
        bronze_dir: Path,
        silver_dir: Path,
        tile_size: int = 640,
        overlap: float = 0.3,
        target_long_side: int | None = 1280,
        skip_existing: bool = True,
        max_images: int | None = None,
        retries: int = 2,
        dedup_threshold: float = 0.85,
        debug_dir: Path | None = None,
        gate: FlyerGate | None = None,
        product_catalog: ProductCatalog | None = None,
        brand_catalog: BrandCatalog | None = None,
    ) -> None:
        self.backend = backend
        self.bronze_dir = Path(bronze_dir)
        self.silver_dir = Path(silver_dir)
        self.tile_size = tile_size
        self.overlap = overlap
        self.target_long_side = target_long_side
        self.skip_existing = skip_existing
        self.max_images = max_images
        self.retries = retries
        self.dedup_threshold = dedup_threshold
        self.debug_dir = Path(debug_dir) if debug_dir else None
        self.gate = gate or AlwaysPassGate()
        self.catalog = product_catalog or ProductCatalog()
        self.brands = brand_catalog or BrandCatalog()
        self.prompt = build_prompt(self.brands)
        # O SilverWriter do projeto grava o JSON final e converte aliases (REQ -> Requeijão).
        self.writer = SilverWriter(
            self.bronze_dir,
            self.silver_dir,
            f"{backend.name}+tiles",
            product_catalog=self.catalog,
        )
        self.stats = _new_stats()

    # ------------------------------------------------------------------ recortes
    def _read_tile(self, jpeg: bytes, label: str) -> ResultadoExtracao | None:
        for attempt in range(1, self.retries + 2):
            try:
                raw = self.backend.extract(jpeg, self.prompt)
                if self.debug_dir:
                    (self.debug_dir / f"{label}.json.txt").write_text(raw, encoding="utf-8")
                return parse_tile_response(raw)
            except Exception as exc:  # rede, timeout, JSON ilegível...
                log.warning("%s: tentativa %d falhou (%s)", label, attempt, exc)
                time.sleep(min(2 * attempt, 6))
        return None

    def _read_image(self, image_path: Path) -> tuple[list[Any], list[str | None]]:
        tiles, (rows, cols) = split_image(
            image_path,
            tile_size=self.tile_size,
            overlap=self.overlap,
            target_long_side=self.target_long_side,
        )
        width = max(t.box[2] for t in tiles)
        height = max(t.box[3] for t in tiles)
        log.info("Imagem %dx%d px -> %d recortes (%dx%d)", width, height, len(tiles), rows, cols)

        if self.debug_dir:
            self.debug_dir.mkdir(parents=True, exist_ok=True)

        items: list[dict[str, Any]] = []
        validades: list[str | None] = []
        failures = 0
        brands_filled = 0

        for tile in tiles:
            label = f"{image_path.stem}_t{tile.index:02d}"
            jpeg = tile.to_jpeg_bytes()
            if self.debug_dir:
                (self.debug_dir / f"{label}.jpg").write_bytes(jpeg)

            log.info("Lendo recorte %d/%d...", tile.index + 1, len(tiles))
            started = time.monotonic()
            result = self._read_tile(jpeg, label)
            log.info("Recorte %d/%d lido em %.0fs", tile.index + 1, len(tiles),
                     time.monotonic() - started)
            if result is None:
                failures += 1
                continue

            brands_filled += fill_missing_brands(result, self.brands)
            validades.append(result.validade)
            if result.encarte_valido:
                items.extend({"oferta": offer, "tile": tile.index} for offer in result.ofertas)

        if failures == len(tiles):
            raise RuntimeError("todos os recortes falharam")
        if failures:
            log.warning("%d de %d recortes falharam; o resultado pode estar incompleto",
                        failures, len(tiles))
        if brands_filled:
            log.info("Marca completada pela lista de marcas em %d produto(s).", brands_filled)

        merged = merge_offers(items, threshold=self.dedup_threshold)
        log.info("Ofertas lidas: %d | após unir recortes vizinhos: %d", len(items), len(merged))
        return merged, validades

    # ------------------------------------------------------------------ imagem
    def process_image(self, image_path: Path) -> bool:
        """Processa uma imagem e grava o JSON. True se for encarte válido."""
        profile = image_path.parent.name

        decision = self.gate.check(image_path)
        if not decision.passed:
            log.info("Imagem ignorada antes da chamada ao modelo (%s).", decision.reason)
            self.writer.save_rejected(image_path, profile, decision.reason or "reprovada_no_filtro")
            return False
        found = list(decision.hits)

        offers, validades = self._read_image(image_path)
        if not offers:
            self.writer.save_rejected(image_path, profile, "modelo_nao_confirmou_encarte", found)
            log.info("Não é encarte. Pulando para a próxima.")
            return False

        result = ResultadoExtracao(
            encarte_valido=True,
            validade=most_common(validades),
            ofertas=offers,
        )
        log.info("Encarte confirmado: %d ofertas | validade: %s",
                 len(offers), result.validade or "não informada")

        lines = self.writer.expand_offers(result, self.stats)
        if not lines:
            self.writer.save_rejected(image_path, profile, "nenhuma_oferta_valida", found)
            log.info("Nenhuma oferta válida (sem preço). Tratado como não encarte.")
            return False

        monitored = set(self.catalog.products)
        found_products = list(dict.fromkeys(
            [*found, *(line["produto"] for line in lines if line["produto"] in monitored)]
        ))
        self.writer.save_result(image_path, profile, result, lines, found_products)
        return True

    # ------------------------------------------------------------------ pasta
    def process_bronze_folder(self) -> dict[str, int]:
        healthcheck = getattr(self.backend, "healthcheck", None)
        if healthcheck:
            healthcheck()

        images = sorted(
            p for p in self.bronze_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
            and not p.stem.lower().endswith(("_ocr", ".tmp"))
        )
        log.info("Imagens encontradas na Bronze: %d", len(images))

        if self.skip_existing:
            pending = [p for p in images if not self.writer.json_path(p).exists()]
            log.info("Já processadas: %d | Pendentes: %d", len(images) - len(pending), len(pending))
            images = pending
        if self.max_images is not None:
            images = images[: self.max_images]
            log.info("Limitando a %d imagem(ns) (max_images).", len(images))
        if not images:
            log.info("Nada novo para processar.")
            return {"encartes": 0, "rejeitadas": 0, "falhas": 0}

        self.stats = _new_stats()
        encartes = rejeitadas = falhas = 0
        durations: list[float] = []
        batch_start = time.monotonic()

        for index, image_path in enumerate(images, start=1):
            started = time.monotonic()
            log.info("=" * 60)
            log.info("[%d/%d] %s | perfil: %s", index, len(images),
                     image_path.name, image_path.parent.name)
            try:
                if self.process_image(image_path):
                    encartes += 1
                else:
                    rejeitadas += 1
            except Exception:
                falhas += 1
                log.exception("FALHA ao processar a imagem %s", image_path)

            durations.append(time.monotonic() - started)
            average = sum(durations) / len(durations)
            remaining = len(images) - index
            log.info("Tempo desta imagem: %.1fs | média: %.1fs | restantes: %d | ETA ≈ %.1f min",
                     durations[-1], average, remaining, average * remaining / 60)

        log.info("=" * 60)
        log.info(
            "RESUMO: %d encartes | %d rejeitadas | %d falhas | %d ofertas | "
            "%d produtos | %d avisos | %.1f min",
            encartes, rejeitadas, falhas, self.stats["ofertas"], self.stats["produtos"],
            self.stats["avisos"], (time.monotonic() - batch_start) / 60,
        )
        return {"encartes": encartes, "rejeitadas": rejeitadas, "falhas": falhas}